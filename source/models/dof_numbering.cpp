#include "../../include/models/dof_numbering.hpp"

template <int dim, int spacedim>
nargil::dof_numbering<dim, spacedim>::dof_numbering()
{
}

//
//

template <int dim, int spacedim>
nargil::dof_numbering<dim, spacedim>::~dof_numbering()
{
}

//
//
//
//
//

template <int dim, int spacedim>
nargil::implicit_hybridized_dof_numbering<dim, spacedim>::
  implicit_hybridized_dof_numbering()
  : dof_numbering<dim, spacedim>()
{
  std::cout << "constructor of implicit_HDG_dof_numbering" << std::endl;
}

template <int dim, int spacedim>
nargil::implicit_hybridized_dof_numbering<dim, spacedim>::
  ~implicit_hybridized_dof_numbering()
{
}

template <int dim, int spacedim>
nargil::model_options::options
nargil::implicit_hybridized_dof_numbering<dim, spacedim>::get_options()
{
  return (model_options::options)(model_options::implicit_time_integration |
                                  model_options::hybridized_dof_numbering);
}

template <int dim, int spacedim>
template <typename ModelEq, typename ModelType>
void nargil::implicit_hybridized_dof_numbering<dim, spacedim>::count_globals(
  ModelType *my_model)
{
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
  for (std::unique_ptr<cell<dim, spacedim> > &cell : my_model->all_owned_cells)
  {
    auto the_worker = static_cast<ModelEq *>(cell.get())->get_worker();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      //      if (cell->dofs_ID_in_this_rank[i_face].size() == 0)
      //      {
      /*
      const auto &face_i1 = cell->dealii_cell->face(i_face);
      //
      // The basic case corresponds to face_i1 being on the boundary.
      // In this case we only need to set the number of current face,
      // and we do not bother to know what is going on, on the other
      // side of this face.  You might wonder why I am not thinking
      // about the case that GenericCell::BCs are set equal to
      // GenericCell::essential here. The reason is that inside the
      // assignment function here, we assign dof numbers to those
      // dofs that have some dof_names for themselves.
      //
      if (face_i1->at_boundary())
      {
        if (cell->BCs[i_face] != GenericCell<dim>::periodic)
        {
          cell->assign_local_global_cell_data(i_face,
                                              local_dof_num_on_this_rank,
                                              global_dof_num_on_this_rank,
                                              this->comm_rank,
                                              0);
          local_dof_num_on_this_rank +=
            cell->dof_names_on_faces[i_face].count();
          global_dof_num_on_this_rank +=
            cell->dof_names_on_faces[i_face].count();
        }
        if (cell->BCs[i_face] == GenericCell<dim>::periodic)
        {
        }
      }
      */
      /*
      else
      {
        //
        // At this point, we are sure that the cell has a neighbor. We
 will
        // have three cases:
        //
        // 1- The neighbor is coarser than the cell. This can only
 happen if
        //    the neighbor is a ghost cell, otherwise there is
 something
        //    wrong. So, when the neighbor is ghost, this subdomain
 does not
        //    own the face. Hence, we have to take the face number
 from the
        //    corresponding neighboer.
        //
        // 2- The neighbor is finer. In this case the face is owned by
 this
        //    subdomain, but we will have two subcases:
        //   2a- If the neighbor is in this subdomain, we act as if
 the domain
        //       was not decomposed.
        //   2b- If the neighbor is in some other subdomain, we have
 to also
        //       send the face number to all those finer neighbors,
 along with
        //       the corresponding subface id.
        //
        // 3- The face has neighbors of same refinement. This case is
 somehow
        //    trichier than what is looks. Because, you have to decide
 where
        //    face belongs to. As we said before, the face belongs to
 the
        //    domain which has smaller rank. So, we have to send the
 face
        //    number from the smaller rank to the higher rank.
        //
        if (cell->dealii_cell->neighbor_is_coarser(i_face))
        {
          //
          // The neighbor should be a ghost, because in each
 subdomain, the
          // elements are ordered from coarse to fine.
          //
          dealiiCell &&nb_i1 = cell->dealii_cell->neighbor(i_face);
          assert(nb_i1->is_ghost());
          unsigned face_nb_num =
 cell->dealii_cell->neighbor_face_no(i_face);
          const auto &face_nb = nb_i1->face(face_nb_num);
          //
          // \bug I believe, nb_face_of_nb_num = i_face. Otherwise,
 something
          // is wrong. I do not change it now, but I will do later.
          //
          unsigned nb_face_of_nb_num =
 nb_i1->neighbor_face_no(face_nb_num);
          for (unsigned i_nb_subface = 0;
               i_nb_subface < face_nb->n_children();
               ++i_nb_subface)
          {
            const dealiiCell &nb_of_nb_i1 =
              nb_i1->neighbor_child_on_subface(face_nb_num,
 i_nb_subface);
            if (nb_of_nb_i1->subdomain_id() == this->comm_rank)
            {
              unsigned nb_of_nb_num =
 this->manager->cell_id_to_num_finder(
                nb_of_nb_i1, this->manager->cell_ID_to_num);
              all_owned_cells[nb_of_nb_num]->assign_local_cell_data(
                nb_face_of_nb_num,
                local_dof_num_on_this_rank,
                nb_i1->subdomain_id(),
                i_nb_subface + 1);
              face_to_rank_recver[nb_i1->subdomain_id()]++;
              if (!is_there_a_msg_from_rank[nb_i1->subdomain_id()])
                is_there_a_msg_from_rank[nb_i1->subdomain_id()] =
 true;
              ++mpi_status_counter;
            }
          }
          local_dof_num_on_this_rank +=
            cell->dof_names_on_faces[i_face].count();
        }
        else if (face_i1->has_children())
        {
          cell->assign_local_global_cell_data(i_face,
                                              local_dof_num_on_this_rank,
                                              global_dof_num_on_this_rank,
                                              this->comm_rank,
                                              0);
          for (unsigned i_subface = 0;
               i_subface < face_i1->number_of_children();
               ++i_subface)
          {
            dealiiCell &&nb_i1 =
              cell->dealii_cell->neighbor_child_on_subface(i_face,
 i_subface);
            int face_nb_i1 =
 cell->dealii_cell->neighbor_face_no(i_face);
            std::stringstream nb_ss_id;
            nb_ss_id << nb_i1->id();
            std::string nb_str_id = nb_ss_id.str();
            if (nb_i1->subdomain_id() == this->comm_rank)
            {
              assert(this->manager->cell_ID_to_num.find(nb_str_id) !=
                     this->manager->cell_ID_to_num.end());
              int nb_i1_num =
 this->manager->cell_ID_to_num[nb_str_id];
              all_owned_cells[nb_i1_num]->assign_local_global_cell_data(
                face_nb_i1,
                local_dof_num_on_this_rank,
                global_dof_num_on_this_rank,
                this->comm_rank,
                i_subface + 1);
            }
            else
            {
              //
              // Here, we are sure that the face is not owned by this
 rank.
              // Also, we know our cell is coarser than nb_i1.
              // Hence, we do not bother to know if the rank of
 neighbor
              // subdomain is greater or smaller than the current
 rank.
              //
              assert(nb_i1->is_ghost());
              assert(ghost_ID_to_num.find(nb_str_id) !=
                     ghost_ID_to_num.end());
              unsigned nb_i1_num = ghost_ID_to_num[nb_str_id];
              all_ghost_cells[nb_i1_num]->assign_local_global_cell_data(
                face_nb_i1,
                local_dof_num_on_this_rank,
                global_dof_num_on_this_rank,
                this->comm_rank,
                i_subface + 1);
              //
              // Now we send id, face id, subface id, and neighbor
 face number
              // to the corresponding rank.
              //
              char buffer[300];
              std::snprintf(buffer,
                            300,
                            "%s#%d#%d#%d",
                            nb_str_id.c_str(),
                            face_nb_i1,
                            i_subface + 1,
                            global_dof_num_on_this_rank);
              face_to_rank_sender[nb_i1->subdomain_id()].push_back(buffer);
              ++mpi_request_counter;
            }
          }
          local_dof_num_on_this_rank +=
            cell->dof_names_on_faces[i_face].count();
          global_dof_num_on_this_rank +=
            cell->dof_names_on_faces[i_face].count();
        }
        else
        {
          dealiiCell &&nb_i1 = cell->dealii_cell->neighbor(i_face);
          int face_nb_i1 =
 cell->dealii_cell->neighbor_face_no(i_face);
          std::stringstream nb_ss_id;
          nb_ss_id << nb_i1->id();
          std::string nb_str_id = nb_ss_id.str();
          if (nb_i1->subdomain_id() == this->comm_rank)
          {
            assert(this->manager->cell_ID_to_num.find(nb_str_id) !=
                   this->manager->cell_ID_to_num.end());
            int nb_i1_num = this->manager->cell_ID_to_num[nb_str_id];
            cell->assign_local_global_cell_data(i_face,
                                                local_dof_num_on_this_rank,
                                                global_dof_num_on_this_rank,
                                                this->comm_rank,
                                                0);
            all_owned_cells[nb_i1_num]->assign_local_global_cell_data(
              face_nb_i1,
              local_dof_num_on_this_rank,
              global_dof_num_on_this_rank,
              this->comm_rank,
              0);
            global_dof_num_on_this_rank +=
              cell->dof_names_on_faces[i_face].count();
          }
          else
          {
            assert(nb_i1->is_ghost());
            if (nb_i1->subdomain_id() > this->comm_rank)
            {
              cell->assign_local_global_cell_data(i_face,
                                                  local_dof_num_on_this_rank,
                                                  global_dof_num_on_this_rank,
                                                  this->comm_rank,
                                                  0);
              assert(ghost_ID_to_num.find(nb_str_id) !=
                     ghost_ID_to_num.end());
              unsigned nb_i1_num = ghost_ID_to_num[nb_str_id];
              all_ghost_cells[nb_i1_num]->assign_local_global_cell_data(
                face_nb_i1,
                local_dof_num_on_this_rank,
                global_dof_num_on_this_rank,
                this->comm_rank,
                0);
              //
              // Now we send id, face id, subface(=0), and neighbor
 face
              // number to the corresponding rank.
              //
              char buffer[300];
              std::snprintf(buffer,
                            300,
                            "%s#%d#%d#%d",
                            nb_str_id.c_str(),
                            face_nb_i1,
                            0,
                            global_dof_num_on_this_rank);
              face_to_rank_sender[nb_i1->subdomain_id()].push_back(buffer);
              global_dof_num_on_this_rank +=
                cell->dof_names_on_faces[i_face].count();
              ++mpi_request_counter;
            }
            else
            {
              cell->assign_local_cell_data(
                i_face, local_dof_num_on_this_rank,
 nb_i1->subdomain_id(), 0);
              face_to_rank_recver[nb_i1->subdomain_id()]++;
              if (!is_there_a_msg_from_rank[nb_i1->subdomain_id()])
                is_there_a_msg_from_rank[nb_i1->subdomain_id()] =
 true;
              ++mpi_status_counter;
            }
          }
          local_dof_num_on_this_rank +=
            cell->dof_names_on_faces[i_face].count();
        }
      }
*/
      //      }
    }
  }

  static_cast<ModelEq *>(my_model->all_owned_cells[0].get())
    ->get_relevant_dofs_count(0);
}
