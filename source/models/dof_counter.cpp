#include "../../include/models/dof_counter.hpp"

template <int dim, int spacedim>
nargil::dof_counter<dim, spacedim>::dof_counter()
{
}

//
//

template <int dim, int spacedim>
nargil::dof_counter<dim, spacedim>::~dof_counter()
{
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::implicit_hybridized_numbering<dim,
                                      spacedim>::implicit_hybridized_numbering()
  : dof_counter<dim, spacedim>()
{
  std::cout << "constructor of implicit_HDG_dof_counter" << std::endl;
}

//
//

template <int dim, int spacedim>
nargil::implicit_hybridized_numbering<dim, spacedim>::
  ~implicit_hybridized_numbering()
{
}

//
//

template <int dim, int spacedim>
template <typename ModelEq, typename ModelType>
void nargil::implicit_hybridized_numbering<dim, spacedim>::count_globals(
  ModelType *my_model)
{
  int comm_rank;
  const MPI_Comm *my_comm = my_model->my_mesh->my_comm;
  MPI_Comm_rank(*my_comm, &comm_rank);

  std::map<unsigned, std::vector<std::string> > face_to_rank_sender;
  std::map<unsigned, unsigned> face_to_rank_recver;
  unsigned local_dof_num_on_this_rank = 0;
  unsigned global_dof_num_on_this_rank = 0;
  unsigned mpi_request_counter = 0;
  unsigned mpi_status_counter = 0;
  std::map<unsigned, bool> is_there_a_msg_from_rank;

  //
  //   <b>Notes for developer 1:</b>
  //   Here, we also count the faces of the model in innerCPU and interCPU
  //   spaces, which are connected to the cells owned by this rank. By
  //   innerCPU, we mean those faces counted in the subdomain of current rank.
  //   By interCPU, we mean those faces which are counted as parts of other
  //   subdomains. We have two rules for this:
  //
  //     - rule 1: If a face is common between two subdomains, and one side is
  //           coarser than the other side. This face belongs to the coarser
  //           side; no matter which subdomain has smaller rank.
  //     - rule 2: If a face is connected to two elements of the same refinement
  //           level, and the elements are in two different subdomains, then
  //           the face belongs to the subdomain with smaller rank.
  //
  for (std::unique_ptr<cell<dim, spacedim> > &i_cell :
       my_model->all_owned_cells)
  {
    auto i_cell_manager = static_cast<ModelEq *>(i_cell.get())
                            ->template get_manager<cell_manager_type>();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      if (i_cell_manager->dofs_ID_in_this_rank[i_face].size() == 0)
      {
        const auto &face_i1 = i_cell->dealii_cell->face(i_face);
        //
        // The basic case corresponds to face_i1 being on the boundary.
        // In this case we only need to set the number of current face,
        // and we do not bother to know what is going on, on the other
        // side of this face.  You might wonder why I am not thinking
        // about the case that BCs are set to
        // essential here. The reason is that inside the
        // assignment function here, we assign dof numbers to those
        // dofs that have some dof_names for themselves.
        //
        if (face_i1->at_boundary())
        {
          if (i_cell_manager->BCs[i_face] != boundary_condition::periodic)
          {
            i_cell_manager->assign_local_global_cell_data(
              i_face,
              local_dof_num_on_this_rank,
              global_dof_num_on_this_rank,
              comm_rank,
              0);
            local_dof_num_on_this_rank +=
              i_cell_manager->dof_names_on_faces[i_face].count();
            global_dof_num_on_this_rank +=
              i_cell_manager->dof_names_on_faces[i_face].count();
          }
        }
      }
      else
      {
        //
        // At this point, we are sure that the cell has a neighbor. We will
        // have three cases:
        //
        // 1- The neighbor is coarser than the cell. This can only happen if
        //    the neighbor is a ghost cell, otherwise there is something
        //    wrong. So, when the neighbor is ghost, this subdomain does not
        //    own the face. Hence, we have to take the face number from the
        //    corresponding neighboer.
        //
        // 2- The neighbor is finer. In this case the face is owned by this
        //    subdomain, but we will have two subcases:
        //   2a- If the neighbor is in this subdomain, we act as if the domain
        //       was not decomposed.
        //   2b- If the neighbor is in some other subdomain, we have to also
        //       send the face number to all those finer neighbors, along with
        //       the corresponding subface id.
        //
        // 3- The face has neighbors of same refinement. This case is somehow
        //    trickier than what it looks. Because, you have to decide where
        //    face belongs to. As we said before, the face belongs to the
        //    domain which has smaller rank. So, we have to send the face
        //    number from the smaller rank to the higher rank.
        //
        if (i_cell->dealii_cell->neighbor_is_coarser(i_face))
        {
          //
          // The neighbor should be a ghost, because in each subdomain, the
          // elements are ordered from coarse to fine.
          //
          dealii_cell_type &&nb_i1 = i_cell->dealii_cell->neighbor(i_face);
          assert(nb_i1->is_ghost());
          unsigned face_nb_num = i_cell->dealii_cell->neighbor_face_no(i_face);
          const auto &face_nb = nb_i1->face(face_nb_num);
          //
          // \bug I believe, nb_face_of_nb_num = i_face. Otherwise, something
          // is wrong. I do not change it now, but I will do later.
          //
          unsigned nb_face_of_nb_num = nb_i1->neighbor_face_no(face_nb_num);
          for (unsigned i_nb_subface = 0; i_nb_subface < face_nb->n_children();
               ++i_nb_subface)
          {
            const dealii_cell_type &nb_of_nb_i1 =
              nb_i1->neighbor_child_on_subface(face_nb_num, i_nb_subface);
            if (nb_of_nb_i1->subdomain_id() == comm_rank)
            {
              unsigned nb_of_nb_num =
                my_model->my_mesh->cell_id_to_num_finder(nb_of_nb_i1, true);
              auto nb_of_nb_manager =
                static_cast<ModelEq *>(
                  my_model->all_owned_cells[nb_of_nb_num].get())
                  ->template get_manager<cell_manager_type>();
              nb_of_nb_manager->assign_local_cell_data(
                nb_face_of_nb_num,
                local_dof_num_on_this_rank,
                nb_i1->subdomain_id(),
                i_nb_subface + 1);
              face_to_rank_recver[nb_i1->subdomain_id()]++;
              if (!is_there_a_msg_from_rank[nb_i1->subdomain_id()])
                is_there_a_msg_from_rank[nb_i1->subdomain_id()] = true;
              ++mpi_status_counter;
            }
          }
          local_dof_num_on_this_rank +=
            i_cell_manager->dof_names_on_faces[i_face].count();
        }
      }
    }
  }

  //  static_cast<ModelEq *>(my_model->all_owned_cells[0].get())
  //    ->get_relevant_dofs_count(0);
}
