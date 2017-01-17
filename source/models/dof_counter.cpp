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
nargil::implicit_hybridized_numbering<
  dim, spacedim>::~implicit_hybridized_numbering()
{
}

//
//

template <int dim, int spacedim>
template <typename BasisType, typename ModelEq>
void nargil::implicit_hybridized_numbering<dim, spacedim>::count_globals(
  nargil::model<ModelEq, dim, spacedim> *my_model)
{
  typedef typename BasisType::CellManagerType CellManagerType;
  int comm_rank, comm_size;
  const MPI_Comm *my_comm = my_model->my_mesh->my_comm;
  MPI_Comm_rank(*my_comm, &comm_rank);
  MPI_Comm_size(*my_comm, &comm_size);

  std::map<unsigned, std::vector<std::string> > face_to_rank_sender;
  std::map<unsigned, unsigned> face_to_rank_recver;
  unsigned i_local_dof_on_this_rank = 0;
  unsigned i_global_dof_on_this_rank = 0;
  unsigned i_local_unkn_on_this_rank = 0;
  unsigned i_global_unkn_on_this_rank = 0;
  unsigned mpi_request_counter = 0;
  unsigned mpi_status_counter = 0;
  std::map<unsigned, bool> is_there_a_msg_from_rank;

  //
  //   Notes for developer 1:
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
    auto i_manager = static_cast<ModelEq *>(i_cell.get())
                       ->template get_manager<CellManagerType>();
    auto i_basis =
      static_cast<ModelEq *>(i_cell.get())->template get_basis<BasisType>();
    std::vector<unsigned> n_unkns_per_dofs = i_basis->get_n_unkns_per_dofs();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      if (i_manager->face_is_not_visited(i_face))
      {
        const auto &face_i1 = i_cell->dealii_cell->face(i_face);
        unsigned n_open_dofs_on_this_face =
          i_manager->dof_status_on_faces[i_face].count();
        unsigned n_open_unkns_on_this_face =
          i_manager->get_n_open_unknowns_on_face(i_face, n_unkns_per_dofs);
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
        if (face_i1->at_boundary() &&
            i_manager->BCs[i_face] != boundary_condition::periodic)
        {
          i_manager->set_cell_properties(i_face, comm_rank, 0);
          i_manager->set_owned_unkn_ids(i_face, i_local_unkn_on_this_rank,
                                        i_global_unkn_on_this_rank,
                                        n_unkns_per_dofs);
          i_manager->assign_local_global_cell_data(
            i_face, i_local_dof_on_this_rank, i_global_dof_on_this_rank);
          i_local_dof_on_this_rank += n_open_dofs_on_this_face;
          i_global_dof_on_this_rank += n_open_dofs_on_this_face;
          i_local_unkn_on_this_rank += n_open_unkns_on_this_face;
          i_global_unkn_on_this_rank += n_open_unkns_on_this_face;
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
            dealiiCell &&nb_i1 = i_cell->dealii_cell->neighbor(i_face);
            assert(nb_i1->is_ghost());
            unsigned face_nb_num =
              i_cell->dealii_cell->neighbor_face_no(i_face);
            const auto &face_nb = nb_i1->face(face_nb_num);
            //
            // \bug I believe, nb_face_of_nb_num = i_face. Otherwise, something
            // is wrong. I do not change it now, but I will do later.
            //
            unsigned nb_face_of_nb_num = nb_i1->neighbor_face_no(face_nb_num);
            assert(nb_face_of_nb_num == i_face);
            for (unsigned i_nb_subface = 0;
                 i_nb_subface < face_nb->n_children();
                 ++i_nb_subface)
            {
              const dealiiCell &nb_of_nb_i1 =
                nb_i1->neighbor_child_on_subface(face_nb_num, i_nb_subface);
              if (nb_of_nb_i1->subdomain_id() == comm_rank)
              {
                auto nb_of_nb_manager =
                  my_model->template get_owned_cell_manager<CellManagerType>(
                    nb_of_nb_i1);
                nb_of_nb_manager->set_cell_properties(
                  nb_face_of_nb_num, nb_i1->subdomain_id(), i_nb_subface + 1);
                nb_of_nb_manager->set_local_unkn_ids(nb_face_of_nb_num,
                                                     i_local_unkn_on_this_rank,
                                                     n_unkns_per_dofs);
                nb_of_nb_manager->assign_local_cell_data(
                  nb_face_of_nb_num, i_local_dof_on_this_rank);
                face_to_rank_recver[nb_i1->subdomain_id()]++;
                if (!is_there_a_msg_from_rank[nb_i1->subdomain_id()])
                  is_there_a_msg_from_rank[nb_i1->subdomain_id()] = true;
                ++mpi_status_counter;
              }
            }
            i_local_dof_on_this_rank += n_open_dofs_on_this_face;
            i_local_unkn_on_this_rank += n_open_unkns_on_this_face;
          }
          //
          // The second case is that the neighbor is finer than this cell.
          //
          else if (face_i1->has_children())
          {
            i_manager->set_cell_properties(i_face, comm_rank, 0);
            i_manager->set_owned_unkn_ids(i_face, i_local_unkn_on_this_rank,
                                          i_global_unkn_on_this_rank,
                                          n_unkns_per_dofs);
            i_manager->assign_local_global_cell_data(
              i_face, i_local_dof_on_this_rank, i_global_dof_on_this_rank);
            for (unsigned i_subface = 0;
                 i_subface < face_i1->number_of_children();
                 ++i_subface)
            {
              dealiiCell &&nb_i1 =
                i_cell->dealii_cell->neighbor_child_on_subface(i_face,
                                                               i_subface);
              int face_nb_i1 = i_cell->dealii_cell->neighbor_face_no(i_face);
              std::stringstream nb_ss_id;
              nb_ss_id << nb_i1->id();
              std::string nb_str_id = nb_ss_id.str();
              if (nb_i1->subdomain_id() == comm_rank)
              {
                auto nb_manager =
                  my_model->template get_owned_cell_manager<CellManagerType>(
                    nb_str_id);
                nb_manager->set_cell_properties(face_nb_i1, comm_rank,
                                                i_subface + 1);
                nb_manager->set_owned_unkn_ids(
                  face_nb_i1, i_local_unkn_on_this_rank,
                  i_global_unkn_on_this_rank, n_unkns_per_dofs);
                nb_manager->assign_local_global_cell_data(
                  face_nb_i1, i_local_dof_on_this_rank,
                  i_global_dof_on_this_rank);
              }
              else
              {
                //
                // Here, we are sure that the neighbor is not owned by this
                // rank. Also, we know our cell is coarser than nb_i1.
                // Hence, we do not bother to know if the rank of neighbor
                // subdomain is greater or smaller than the current rank.
                //
                auto nb_manager =
                  my_model->template get_ghost_cell_manager<CellManagerType>(
                    nb_str_id);
                nb_manager->set_cell_properties(face_nb_i1, comm_rank,
                                                i_subface + 1);
                nb_manager->set_owned_unkn_ids(
                  face_nb_i1, i_local_unkn_on_this_rank,
                  i_global_unkn_on_this_rank, n_unkns_per_dofs);
                nb_manager->assign_local_global_cell_data(
                  face_nb_i1, i_local_dof_on_this_rank,
                  i_global_dof_on_this_rank);
                //
                // Now we send id, face id, subface id, and neighbor face number
                // to the corresponding rank.
                //
                char buffer[300];
                std::snprintf(buffer,
                              300,
                              "%s#%d#%d#%d",
                              nb_str_id.c_str(),
                              face_nb_i1,
                              i_subface + 1,
                              i_global_unkn_on_this_rank);
                face_to_rank_sender[nb_i1->subdomain_id()].push_back(buffer);
                ++mpi_request_counter;
              }
            }
            i_local_dof_on_this_rank += n_open_dofs_on_this_face;
            i_global_dof_on_this_rank += n_open_dofs_on_this_face;
            i_local_unkn_on_this_rank += n_open_unkns_on_this_face;
            i_global_unkn_on_this_rank += n_open_unkns_on_this_face;
          }
          //
          // The third case is that the neighbor has the same level of
          // refinement as the current cell.
          //
          else
          {
            dealiiCell &&nb_i1 = i_cell->dealii_cell->neighbor(i_face);
            int face_nb_i1 = i_cell->dealii_cell->neighbor_face_no(i_face);
            std::stringstream nb_ss_id;
            nb_ss_id << nb_i1->id();
            std::string nb_str_id = nb_ss_id.str();
            if (nb_i1->subdomain_id() == comm_rank)
            {
              auto nb_manager =
                my_model->template get_owned_cell_manager<CellManagerType>(
                  nb_str_id);
              i_manager->set_cell_properties(i_face, comm_rank, 0);
              i_manager->set_owned_unkn_ids(i_face, i_local_unkn_on_this_rank,
                                            i_global_unkn_on_this_rank,
                                            n_unkns_per_dofs);
              i_manager->assign_local_global_cell_data(
                i_face, i_local_dof_on_this_rank, i_global_dof_on_this_rank);
              nb_manager->set_cell_properties(face_nb_i1, comm_rank, 0);
              nb_manager->set_owned_unkn_ids(
                face_nb_i1, i_local_unkn_on_this_rank,
                i_global_unkn_on_this_rank, n_unkns_per_dofs);
              nb_manager->assign_local_global_cell_data(
                face_nb_i1, i_local_dof_on_this_rank,
                i_global_dof_on_this_rank);
              i_global_dof_on_this_rank += n_open_dofs_on_this_face;
              i_global_unkn_on_this_rank += n_open_unkns_on_this_face;
            }
            else
            {
              assert(nb_i1->is_ghost());
              if (nb_i1->subdomain_id() > comm_rank)
              {
                i_manager->set_cell_properties(i_face, comm_rank, 0);
                i_manager->set_owned_unkn_ids(i_face, i_local_unkn_on_this_rank,
                                              i_global_unkn_on_this_rank,
                                              n_unkns_per_dofs);
                i_manager->assign_local_global_cell_data(
                  i_face, i_local_dof_on_this_rank, i_global_dof_on_this_rank);
                auto nb_manager =
                  my_model->template get_ghost_cell_manager<CellManagerType>(
                    nb_str_id);
                nb_manager->set_cell_properties(face_nb_i1, comm_rank, 0);
                nb_manager->set_owned_unkn_ids(
                  face_nb_i1, i_local_unkn_on_this_rank,
                  i_global_unkn_on_this_rank, n_unkns_per_dofs);
                nb_manager->assign_local_global_cell_data(
                  face_nb_i1, i_local_dof_on_this_rank,
                  i_global_dof_on_this_rank);
                //
                // Now we send id, face id, subface(=0), and neighbor face
                // number to the corresponding rank.
                //
                char buffer[300];
                std::snprintf(buffer,
                              300,
                              "%s#%d#%d#%d",
                              nb_str_id.c_str(),
                              face_nb_i1,
                              0,
                              i_global_unkn_on_this_rank);
                face_to_rank_sender[nb_i1->subdomain_id()].push_back(buffer);
                i_global_dof_on_this_rank += n_open_dofs_on_this_face;
                i_global_unkn_on_this_rank += n_open_unkns_on_this_face;
                ++mpi_request_counter;
              }
              else
              {
                i_manager->set_cell_properties(i_face, nb_i1->subdomain_id(),
                                               0);
                i_manager->set_local_unkn_ids(i_face, i_local_unkn_on_this_rank,
                                              n_unkns_per_dofs);
                i_manager->assign_local_cell_data(i_face,
                                                  i_local_dof_on_this_rank);
                face_to_rank_recver[nb_i1->subdomain_id()]++;
                if (!is_there_a_msg_from_rank[nb_i1->subdomain_id()])
                  is_there_a_msg_from_rank[nb_i1->subdomain_id()] = true;
                ++mpi_status_counter;
              }
            }
            i_local_dof_on_this_rank += n_open_dofs_on_this_face;
            i_local_unkn_on_this_rank += n_open_unkns_on_this_face;
          }
        }
      }
    }
  }

  printf("%d %d %d %d\n", i_local_dof_on_this_rank, i_global_dof_on_this_rank,
         i_local_unkn_on_this_rank, i_global_unkn_on_this_rank);

  //
  // The next two variables contain num faces from rank zero to the
  // current rank, including and excluding current rank
  //
  std::vector<unsigned> dofs_count_be4_rank(comm_size, 0);
  std::vector<unsigned> dofs_count_up2_rank(comm_size, 0);
  std::vector<unsigned> unkns_count_be4_rank(comm_size, 0);
  std::vector<unsigned> unkns_count_up2_rank(comm_size, 0);
  unsigned n_dofs_this_rank_owns = i_global_dof_on_this_rank;
  unsigned n_unkns_this_rank_owns = i_global_unkn_on_this_rank;

  MPI_Allgather(&n_dofs_this_rank_owns,
                1,
                MPI_UNSIGNED,
                dofs_count_up2_rank.data(),
                1,
                MPI_UNSIGNED,
                *my_comm);

  MPI_Allgather(&n_unkns_this_rank_owns,
                1,
                MPI_UNSIGNED,
                unkns_count_up2_rank.data(),
                1,
                MPI_UNSIGNED,
                *my_comm);

  for (unsigned i_num = 0; i_num < comm_size; ++i_num)
    for (unsigned j_num = 0; j_num < i_num; ++j_num)
    {
      dofs_count_be4_rank[i_num] += dofs_count_up2_rank[j_num];
      unkns_count_be4_rank[i_num] += unkns_count_up2_rank[j_num];
    }

  for (std::unique_ptr<cell<dim> > &i_cell : my_model->all_owned_cells)
  {
    auto i_manager = static_cast<ModelEq *>(i_cell.get())
                       ->template get_manager<CellManagerType>();
    i_manager->offset_global_unkn_ids(unkns_count_be4_rank[comm_rank]);
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
      for (unsigned i_dof = 0;
           i_dof < i_manager->dofs_ID_in_all_ranks[i_face].size();
           ++i_dof)
        i_manager->dofs_ID_in_all_ranks[i_face][i_dof] +=
          dofs_count_be4_rank[comm_rank];
  }

  for (std::unique_ptr<cell<dim> > &ghost_cell : my_model->all_ghost_cells)
  {
    auto ghost_manager = static_cast<ModelEq *>(ghost_cell.get())
                           ->template get_manager<CellManagerType>();
    ghost_manager->offset_global_unkn_ids(unkns_count_be4_rank[comm_rank]);
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
      for (unsigned i_dof = 0;
           i_dof < ghost_manager->dofs_ID_in_all_ranks[i_face].size();
           ++i_dof)
        ghost_manager->dofs_ID_in_all_ranks[i_face][i_dof] +=
          dofs_count_be4_rank[comm_rank];
  }

  //
  // Now, we want to also assign a unique number to those faces which
  // do not belong to the current rank. This includes those faces which
  // are connected to a locally owned cell or not. We start ghost face
  // numbers from -10, and go down.
  //
  int ghost_dofs_counter = -10;
  int ghost_unkns_counter = -10;
  for (std::unique_ptr<cell<dim> > &ghost_cell : my_model->all_ghost_cells)
  {
    auto ghost_manager = static_cast<ModelEq *>(ghost_cell.get())
                           ->template get_manager<CellManagerType>();
    auto i_basis =
      static_cast<ModelEq *>(ghost_cell.get())->template get_basis<BasisType>();
    std::vector<unsigned> n_unkns_per_dofs = i_basis->get_n_unkns_per_dofs();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      unsigned n_open_dofs_on_this_face =
        ghost_manager->dof_status_on_faces[i_face].count();
      unsigned n_open_unkns_on_this_face =
        ghost_manager->get_n_open_unknowns_on_face(i_face, n_unkns_per_dofs);
      if (ghost_manager->face_is_not_visited(i_face))
      {
        const auto &face_i1 = ghost_cell->dealii_cell->face(i_face);
        //
        // The basic case corresponds to face_i1 being on the boundary.
        // In this case we only need to set the number of current face,
        // and we do not bother to know what is going on, on the other
        // side of this face.
        //
        if (face_i1->at_boundary())
        {
          {
            ghost_manager->set_cell_properties(
              i_face, ghost_cell->dealii_cell->subdomain_id(), 0);
            ghost_manager->set_ghost_unkn_ids(i_face, ghost_unkns_counter,
                                              n_unkns_per_dofs);
            ghost_manager->assign_ghost_cell_data(i_face, ghost_dofs_counter);
            ghost_dofs_counter -= n_open_dofs_on_this_face;
            ghost_unkns_counter -= n_open_unkns_on_this_face;
          }
        }
        else
        {
          //
          // We are sure that the face that we are on, is either on the coarser
          // side of an owned cell, or belongs to a lower rank than the current
          // rank.
          //
          ghost_manager->set_cell_properties(
            i_face, ghost_cell->dealii_cell->subdomain_id(), 0);
          ghost_manager->set_ghost_unkn_ids(i_face, ghost_unkns_counter,
                                            n_unkns_per_dofs);
          ghost_manager->assign_ghost_cell_data(i_face, ghost_dofs_counter);
          if (face_i1->has_children())
          {
            int face_nb_subface =
              ghost_cell->dealii_cell->neighbor_face_no(i_face);
            for (unsigned i_subface = 0;
                 i_subface < face_i1->number_of_children();
                 ++i_subface)
            {
              dealiiCell &&nb_subface =
                ghost_cell->dealii_cell->neighbor_child_on_subface(i_face,
                                                                   i_subface);
              if (nb_subface->is_ghost())
              {
                auto nb_manager =
                  my_model->template get_ghost_cell_manager<CellManagerType>(
                    nb_subface);
                nb_manager->set_cell_properties(
                  face_nb_subface, nb_subface->subdomain_id(), i_subface + 1);
                nb_manager->set_ghost_unkn_ids(
                  face_nb_subface, ghost_unkns_counter, n_unkns_per_dofs);
                nb_manager->assign_ghost_cell_data(face_nb_subface,
                                                   ghost_dofs_counter);
              }
            }
          }
          else if (ghost_cell->dealii_cell->neighbor(i_face)->is_ghost())
          {
            dealiiCell &&nb_i1 = ghost_cell->dealii_cell->neighbor(i_face);
            int face_nb_i1 = ghost_cell->dealii_cell->neighbor_face_no(i_face);
            auto nb_manager =
              my_model->template get_ghost_cell_manager<CellManagerType>(nb_i1);
            assert(nb_manager->dofs_ID_in_this_rank[face_nb_i1].size() == 0);
            assert(nb_manager->dofs_ID_in_all_ranks[face_nb_i1].size() == 0);
            nb_manager->set_cell_properties(face_nb_i1, nb_i1->subdomain_id(),
                                            0);
            nb_manager->set_ghost_unkn_ids(face_nb_i1, ghost_unkns_counter,
                                           n_unkns_per_dofs);
            nb_manager->assign_ghost_cell_data(face_nb_i1, ghost_dofs_counter);
          }
          ghost_dofs_counter -= n_open_dofs_on_this_face;
          ghost_unkns_counter -= n_open_unkns_on_this_face;
        }
      }
    }
  }

  //
  // Here we start the interCPU communications. According to almost
  // every specification, to avoid high communication overhead, we
  // should perform all send/recv's in one go. To gain more control, we
  // do this process in two phases:
  //
  //   - Phase 1: We are sending from lower ranks to higher ranks.
  //              Hence, higher ranks skip the sending loop and lower
  //              ranks skip the recv loop.
  //   - Phase 2: We send from higher ranks to lower ranks. Hence,
  //              the lower ranks will skip the sending loop.
  //
  // This way, we make sure that no deadlock will happen.
  //

  //
  // Phase 1 : Send Loop
  //
  for (auto &&i_send = face_to_rank_sender.rbegin();
       i_send != face_to_rank_sender.rend();
       ++i_send)
  {
    assert(comm_rank != i_send->first);
    if (comm_rank < i_send->first)
    {
      unsigned num_sends = face_to_rank_sender[i_send->first].size();
      unsigned jth_rank_on_i_send = 0;
      std::vector<MPI_Request> all_mpi_reqs_of_rank(num_sends);
      for (auto &&msg_it : face_to_rank_sender[i_send->first])
      {
        MPI_Isend((char *)msg_it.c_str(),
                  msg_it.size() + 1,
                  MPI_CHAR,
                  i_send->first,
                  my_model->my_mesh->refn_cycle,
                  *my_comm,
                  &all_mpi_reqs_of_rank[jth_rank_on_i_send]);
        ++jth_rank_on_i_send;
      }
      MPI_Waitall(num_sends, all_mpi_reqs_of_rank.data(), MPI_STATUSES_IGNORE);
    }
  }

  //
  // Phase 1 : Recv Loop
  //
  std::vector<MPI_Status> all_mpi_stats_of_rank(mpi_status_counter);
  unsigned recv_counter = 0;
  bool no_msg_left = (is_there_a_msg_from_rank.size() == 0);
  while (!no_msg_left)
  {
    auto i_recv = is_there_a_msg_from_rank.begin();
    no_msg_left = true;
    for (; i_recv != is_there_a_msg_from_rank.end(); ++i_recv)
    {
      if (i_recv->second && comm_rank > i_recv->first)
        no_msg_left = false;
      int flag = 0;
      if (comm_rank > i_recv->first)
        MPI_Iprobe(i_recv->first,
                   my_model->my_mesh->refn_cycle,
                   *my_comm,
                   &flag,
                   MPI_STATUS_IGNORE);
      if (flag)
      {
        assert(i_recv->second);
        break;
      }
    }
    if (i_recv != is_there_a_msg_from_rank.end())
    {
      for (unsigned jth_rank_on_i_recv = 0;
           jth_rank_on_i_recv < face_to_rank_recver[i_recv->first];
           ++jth_rank_on_i_recv)
      {
        char buffer[300];
        MPI_Recv(&buffer[0],
                 300,
                 MPI_CHAR,
                 i_recv->first,
                 my_model->my_mesh->refn_cycle,
                 *my_comm,
                 &all_mpi_stats_of_rank[recv_counter]);
        std::vector<std::string> tokens;
        Tokenize(buffer, tokens, "#");
        assert(tokens.size() == 4);
        std::string cell_unique_id = tokens[0];
        unsigned face_num = std::stoi(tokens[1]);
        auto i_manager =
          my_model->template get_owned_cell_manager<CellManagerType>(
            cell_unique_id);
        auto i_basis =
          my_model->template get_owned_cell_basis<BasisType>(cell_unique_id);
        std::vector<unsigned> n_unkns_per_dofs =
          i_basis->get_n_unkns_per_dofs();
        assert(i_manager->dofs_ID_in_all_ranks[face_num].size() == 0);
        assert(i_manager->dof_status_on_faces[face_num].count() != 0);
        //
        // The DOF data received from other CPU is the ID of the first
        // DOF of this face.
        //
        i_manager->set_nonlocal_unkn_ids(
          face_num, std::stoi(tokens[3]) + unkns_count_be4_rank[i_recv->first],
          n_unkns_per_dofs);
        /*
        for (unsigned i_dof = 0;
             i_dof < i_manager->dof_status_on_faces[face_num].count();
             ++i_dof)
          i_manager->dofs_ID_in_all_ranks[face_num].push_back(
            std::stoi(tokens[3]) + dofs_count_be4_rank[i_recv->first] + i_dof);
        */
        ++recv_counter;
      }
      i_recv->second = false;
    }
  }

  //
  // Phase 2 : Send Loop
  //
  for (auto &&i_send = face_to_rank_sender.rbegin();
       i_send != face_to_rank_sender.rend();
       ++i_send)
  {
    assert(comm_rank != i_send->first);
    if (comm_rank > i_send->first)
    {
      unsigned num_sends = face_to_rank_sender[i_send->first].size();
      unsigned jth_rank_on_i_send = 0;
      std::vector<MPI_Request> all_mpi_reqs_of_rank(num_sends);
      for (auto &&msg_it : face_to_rank_sender[i_send->first])
      {
        MPI_Isend((char *)msg_it.c_str(),
                  msg_it.size() + 1,
                  MPI_CHAR,
                  i_send->first,
                  my_model->my_mesh->refn_cycle,
                  *my_comm,
                  &all_mpi_reqs_of_rank[jth_rank_on_i_send]);
        ++jth_rank_on_i_send;
      }
      MPI_Waitall(num_sends, all_mpi_reqs_of_rank.data(), MPI_STATUSES_IGNORE);
    }
  }

  //
  // Phase 2 : Recv Loop
  //
  no_msg_left = (is_there_a_msg_from_rank.size() == 0);
  while (!no_msg_left)
  {
    auto i_recv = is_there_a_msg_from_rank.begin();
    no_msg_left = true;
    for (; i_recv != is_there_a_msg_from_rank.end(); ++i_recv)
    {
      if (i_recv->second && comm_rank < i_recv->first)
        no_msg_left = false;
      int flag = 0;
      if (comm_rank < i_recv->first)
        MPI_Iprobe(i_recv->first,
                   my_model->my_mesh->refn_cycle,
                   *my_comm,
                   &flag,
                   MPI_STATUS_IGNORE);
      if (flag)
      {
        assert(i_recv->second);
        break;
      }
    }
    if (i_recv != is_there_a_msg_from_rank.end())
    {
      for (unsigned jth_rank_on_i_recv = 0;
           jth_rank_on_i_recv < face_to_rank_recver[i_recv->first];
           ++jth_rank_on_i_recv)
      {
        char buffer[300];
        MPI_Recv(&buffer[0],
                 300,
                 MPI_CHAR,
                 i_recv->first,
                 my_model->my_mesh->refn_cycle,
                 *my_comm,
                 &all_mpi_stats_of_rank[recv_counter]);
        std::vector<std::string> tokens;
        Tokenize(buffer, tokens, "#");
        assert(tokens.size() == 4);
        std::string cell_unique_id = tokens[0];
        unsigned face_num = std::stoi(tokens[1]);
        auto i_manager =
          my_model->template get_owned_cell_manager<CellManagerType>(
            cell_unique_id);
        auto i_basis =
          my_model->template get_owned_cell_basis<BasisType>(cell_unique_id);
        std::vector<unsigned> n_unkns_per_dofs =
          i_basis->get_n_unkns_per_dofs();
        assert(i_manager->dof_status_on_faces[face_num].count() != 0);
        assert(i_manager->dofs_ID_in_all_ranks[face_num].size() == 0);
        //
        // The DOF data received from other CPU is the ID of the first
        // DOF of this face.
        //
        i_manager->set_nonlocal_unkn_ids(
          face_num, std::stoi(tokens[3]) + unkns_count_be4_rank[i_recv->first],
          n_unkns_per_dofs);
        /*
        for (unsigned i_dof = 0;
             i_dof < i_manager->dof_status_on_faces[face_num].count();
             ++i_dof)
          i_manager->dofs_ID_in_all_ranks[face_num].push_back(
            std::stoi(tokens[3]) + dofs_count_be4_rank[i_recv->first] + i_dof);
        */
        ++recv_counter;
      }
      i_recv->second = false;
    }
  }

  //
  // Before contnuing with the rest of this function, we deefine a
  // structure which serves as a generic DoF, which stores such things
  // as the parent cell of each DoF, and the corresponding face.
  //

  using CellIter = typename cell<dim, spacedim>::CellIter;
  struct dof_properties
  {
    dof_properties() : n_local_connected_DOFs(0), n_nonlocal_connected_DOFs(0)
    {
    }
    unsigned n_local_connected_DOFs;
    unsigned n_nonlocal_connected_DOFs;
    std::vector<CellIter> parent_cells;
    std::vector<unsigned> connected_face_of_parent_cell;
    std::vector<CellIter> parent_ghosts;
    std::vector<unsigned> connected_face_of_parent_ghost;
  };

  //
  //         THESE NEXT LOOPS ARE JUST FOR THE SOLVER !!
  //
  // When you want to preallocate stiffness matrix in PETSc, it
  // accpet an argument which contains the number of DOFs connected to
  // the DOF in each row. According to PETSc, if you let it know
  // about this preallocation, you will get a noticeable performance
  // boost.
  //
  // Now, we want to know, each face belonging to the current
  // rank is connected to how many faces from the current rank and
  // how many faces from other ranks. So, if for example, we are on
  // rank 1, we want to be able to count the innerFaces and
  // interFaces shown below (This is especilly a challange,
  // because the faces of elements on the right are connected to
  // faces of elements in the left via two middle ghost elements.):
  //
  //               ---------------------------
  //                rank 1 | rank 2 | rank 1
  //               --------|--------|---------
  //                rank 1 | rank 2 | rank 1
  //               ----------------------------
  //
  // To this end, Let us build a vector containing each unique
  // face which belongs to this rank (So those faces which do
  // not belong to this rank are not present in this vector !).
  // Then, fill the GenericFace::Parent_Cells vector with
  // those parent cells which also belongs to the current rank.
  // Also, we fill GenericFace::Parent_Ghosts with ghost cells
  // connected to the current face.
  //
  std::vector<dof_properties> all_owned_dofs(i_global_dof_on_this_rank);
  for (CellIter cell_it = my_model->all_owned_cells.begin();
       cell_it != my_model->all_owned_cells.end();
       ++cell_it)
  {
    auto i_manager = static_cast<ModelEq *>(cell_it->get())
                       ->template get_manager<CellManagerType>();

    /*
    //
    //
    //

    for (unsigned i_face = 0; i_face < (*cell_it)->n_faces; ++i_face)
    {
      if ((*cell_it)->face_owner_rank[i_face] == this->comm_rank)
      {
        for (unsigned i_dof = 0;
             i_dof < (*cell_it)->dofs_ID_in_all_ranks[i_face].size();
             ++i_dof)
        {
          //

          int dof_i1 = (*cell_it)->dofs_ID_in_all_ranks[i_face][i_dof] -
                       dofs_count_be4_rank[this->comm_rank];
          all_owned_dofs[dof_i1].parent_cells.push_back(cell_it);
          all_owned_dofs[dof_i1].connected_face_of_parent_cell.push_back(
            i_face);
          //
          // Here, we just add the open DoFs on the face itself. The reason
          // for this is to avoid
          //
          if (all_owned_dofs[dof_i1].n_local_connected_DOFs == 0)
            all_owned_dofs[dof_i1].n_local_connected_DOFs =
              (*cell_it)->dof_status_on_faces[i_face].count();

          //
        }
      }
    }

    //
    //
    //

    */
  }
  /*
    for (typename GenericCell<dim>::CellIter ghost_cell_it =
           all_ghost_cells.begin();
         ghost_cell_it != all_ghost_cells.end();
         ++ghost_cell_it)
    {
      for (unsigned i_face = 0; i_face < (*ghost_cell_it)->n_faces; ++i_face)
      {
        if ((*ghost_cell_it)->face_owner_rank[i_face] == this->comm_rank)
        {
          for (unsigned i_dof = 0;
               i_dof < (*ghost_cell_it)->dofs_ID_in_all_ranks[i_face].size();
               ++i_dof)
          {
            int dof_i1 = (*ghost_cell_it)->dofs_ID_in_all_ranks[i_face][i_dof] -
                         dofs_count_be4_rank[this->comm_rank];
            all_owned_dofs[dof_i1].parent_ghosts.push_back(ghost_cell_it);
            all_owned_dofs[dof_i1].connected_face_of_parent_ghost.push_back(
              i_face);
            if (all_owned_dofs[dof_i1].n_local_connected_DOFs == 0)
            {
              std::cout << "This is a curious case which should not happen. "
                           "How is that a ghost cell can give a face ownership?"
                        << std::endl;
              assert(all_owned_dofs[dof_i1].n_local_connected_DOFs != 0);
            }
          }
        }
      }
    }

    this->n_global_DOFs_rank_owns = n_dofs_this_rank_owns * n_face_bases;

    //
    //
    //
    for (GenericDOF<dim> &dof : all_owned_dofs)
    {
      std::map<int, unsigned> local_dofs_num_map;
      std::map<int, unsigned> nonlocal_dofs_num_map;
      for (unsigned i_parent_cell = 0; i_parent_cell < dof.parent_cells.size();
           ++i_parent_cell)
      {
        auto parent_cell = dof.parent_cells[i_parent_cell];
        for (unsigned j_face = 0; j_face < n_faces_per_cell; ++j_face)
        {
          unsigned face_ij = dof.connected_face_of_parent_cell[i_parent_cell];
          if (j_face != face_ij)
            for (unsigned i_dof = 0;
                 i_dof < (*parent_cell)->dofs_ID_in_all_ranks[j_face].size();
                 ++i_dof)
            {
              if ((*parent_cell)->face_owner_rank[j_face] == this->comm_rank)
                local_dofs_num_map[(*parent_cell)
                                     ->dofs_ID_in_all_ranks[j_face][i_dof]]++;
              else
                nonlocal_dofs_num_map[(*parent_cell)
                                        ->dofs_ID_in_all_ranks[j_face][i_dof]]++;
            }
        }
      }

      for (unsigned i_parent_ghost = 0; i_parent_ghost <
    dof.parent_ghosts.size();
           ++i_parent_ghost)
      {
        auto parent_ghost = dof.parent_ghosts[i_parent_ghost];
        for (unsigned j_face = 0; j_face < n_faces_per_cell; ++j_face)
        {
          unsigned face_ij = dof.connected_face_of_parent_ghost[i_parent_ghost];
          if (j_face != face_ij)
            for (unsigned i_dof = 0;
                 i_dof < (*parent_ghost)->dof_status_on_faces[j_face].count();
                 ++i_dof)
            {
              if ((*parent_ghost)->face_owner_rank[j_face] == this->comm_rank)
                local_dofs_num_map[(*parent_ghost)
                                     ->dofs_ID_in_all_ranks[j_face][i_dof]]++;
              else
                nonlocal_dofs_num_map[(*parent_ghost)
                                        ->dofs_ID_in_all_ranks[j_face][i_dof]]++;
            }
        }
      }
      dof.n_local_connected_DOFs += local_dofs_num_map.size();
      dof.n_nonlocal_connected_DOFs = nonlocal_dofs_num_map.size();
    }

    MPI_Allreduce(&this->n_global_DOFs_rank_owns,
                  &this->n_global_DOFs_on_all_ranks,
                  1,
                  MPI_UNSIGNED,
                  MPI_SUM,
                  this->manager->comm);

    int dof_counter = 0;
    this->n_local_DOFs_connected_to_DOF.resize(this->n_global_DOFs_rank_owns);
    this->n_nonlocal_DOFs_connected_to_DOF.resize(this->n_global_DOFs_rank_owns);
    for (GenericDOF<dim> &dof : all_owned_dofs)
    {
      for (unsigned unknown = 0; unknown < n_face_bases; ++unknown)
      {
        this->n_local_DOFs_connected_to_DOF[dof_counter + unknown] +=
          dof.n_local_connected_DOFs * n_face_bases;
        this->n_nonlocal_DOFs_connected_to_DOF[dof_counter + unknown] +=
          dof.n_nonlocal_connected_DOFs * n_face_bases;
      }
      dof_counter += n_face_bases;
    }

    std::map<unsigned, unsigned> map_from_local_to_global;
    for (std::unique_ptr<GenericCell<dim> > &cell : all_owned_cells)
    {
      for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
      {
        for (unsigned i_dof = 0;
             i_dof < cell->dofs_ID_in_this_rank[i_face].size();
             ++i_dof)
        {
          int index1 = cell->dofs_ID_in_this_rank[i_face][i_dof];
          int index2 = cell->dofs_ID_in_all_ranks[i_face][i_dof];
          assert(index1 >= 0 && index2 >= 0);
          map_from_local_to_global[index1] = index2;
        }
      }
    }
    assert(map_from_local_to_global.size() == local_dof_num_on_this_rank);

    this->n_local_DOFs_on_this_rank = local_dof_num_on_this_rank * n_face_bases;
    this->scatter_from.reserve(this->n_local_DOFs_on_this_rank);
    this->scatter_to.reserve(this->n_local_DOFs_on_this_rank);
    for (const auto &map_it : map_from_local_to_global)
    {
      for (unsigned i_polyface = 0; i_polyface < n_face_bases; ++i_polyface)
      {
        this->scatter_to.push_back(map_it.first * n_face_bases + i_polyface);
        this->scatter_from.push_back(map_it.second * n_face_bases + i_polyface);
      }
    }

    char buffer[100];
    std::snprintf(buffer,
                  100,
                  "Number of DOFs in this rank is: %d and number of dofs "
                  "in all "
                  "ranks is : %d",
                  this->n_global_DOFs_rank_owns,
                  this->n_global_DOFs_on_all_ranks);
    this->manager->out_logger(this->manager->execution_time, buffer, true);
    */
}
