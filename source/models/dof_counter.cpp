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
  nargil::model<ModelEq, dim, spacedim> *in_model)
{
  typedef typename BasisType::CellManagerType CellManagerType;
  int comm_rank, comm_size;
  const MPI_Comm *my_comm = in_model->my_mesh->my_comm;
  MPI_Comm_rank(*my_comm, &comm_rank);
  MPI_Comm_size(*my_comm, &comm_size);

  std::map<unsigned, std::vector<std::string> > face_to_rank_sender;
  std::map<unsigned, unsigned> face_to_rank_recver;
  unsigned i_local_unkn_on_this_rank = 0;
  unsigned i_global_unkn_on_this_rank = 0;
  unsigned mpi_request_counter = 0;
  unsigned mpi_status_counter = 0;
  std::map<unsigned, bool> is_there_a_msg_from_rank;

  unsigned local_interior_unkn_id = 0;

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
       in_model->all_owned_cells)
  {
    auto i_manager = static_cast<ModelEq *>(i_cell.get())
                       ->template get_manager<CellManagerType>();
    auto i_basis =
      static_cast<ModelEq *>(i_cell.get())->template get_basis<BasisType>();
    i_manager->set_local_interior_unkn_id(&local_interior_unkn_id);
    std::vector<unsigned> n_unkns_per_dofs = i_basis->get_n_unkns_per_dofs();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      if (i_manager->face_is_not_visited(i_face))
      {
        const auto &face_i1 = i_cell->my_dealii_cell->face(i_face);
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
          if (i_cell->my_dealii_cell->neighbor_is_coarser(i_face))
          {
            //
            // The neighbor should be a ghost, because in each subdomain, the
            // elements are ordered from coarse to fine.
            //
            dealiiTriCell &&nb_i1 = i_cell->my_dealii_cell->neighbor(i_face);
            assert(nb_i1->is_ghost());
            unsigned face_nb_num =
              i_cell->my_dealii_cell->neighbor_face_no(i_face);
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
              const dealiiTriCell &nb_of_nb_i1 =
                nb_i1->neighbor_child_on_subface(face_nb_num, i_nb_subface);
              if (nb_of_nb_i1->subdomain_id() == comm_rank)
              {
                auto nb_of_nb_manager =
                  in_model->template get_owned_cell_manager<CellManagerType>(
                    nb_of_nb_i1);
                nb_of_nb_manager->set_cell_properties(
                  nb_face_of_nb_num, nb_i1->subdomain_id(), i_nb_subface + 1);
                nb_of_nb_manager->set_local_unkn_ids(nb_face_of_nb_num,
                                                     i_local_unkn_on_this_rank,
                                                     n_unkns_per_dofs);
                face_to_rank_recver[nb_i1->subdomain_id()]++;
                if (!is_there_a_msg_from_rank[nb_i1->subdomain_id()])
                  is_there_a_msg_from_rank[nb_i1->subdomain_id()] = true;
                ++mpi_status_counter;
              }
            }
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
            for (unsigned i_subface = 0;
                 i_subface < face_i1->number_of_children();
                 ++i_subface)
            {
              dealiiTriCell &&nb_i1 =
                i_cell->my_dealii_cell->neighbor_child_on_subface(i_face,
                                                                  i_subface);
              int face_nb_i1 = i_cell->my_dealii_cell->neighbor_face_no(i_face);
              std::stringstream nb_ss_id;
              nb_ss_id << nb_i1->id();
              std::string nb_str_id = nb_ss_id.str();
              if (nb_i1->subdomain_id() == comm_rank)
              {
                auto nb_manager =
                  in_model->template get_owned_cell_manager<CellManagerType>(
                    nb_str_id);
                nb_manager->set_cell_properties(face_nb_i1, comm_rank,
                                                i_subface + 1);
                nb_manager->set_owned_unkn_ids(
                  face_nb_i1, i_local_unkn_on_this_rank,
                  i_global_unkn_on_this_rank, n_unkns_per_dofs);
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
                  in_model->template get_ghost_cell_manager<CellManagerType>(
                    nb_str_id);
                nb_manager->set_cell_properties(face_nb_i1, comm_rank,
                                                i_subface + 1);
                nb_manager->set_owned_unkn_ids(
                  face_nb_i1, i_local_unkn_on_this_rank,
                  i_global_unkn_on_this_rank, n_unkns_per_dofs);
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
            i_local_unkn_on_this_rank += n_open_unkns_on_this_face;
            i_global_unkn_on_this_rank += n_open_unkns_on_this_face;
          }
          //
          // The third case is that the neighbor has the same level of
          // refinement as the current cell.
          //
          else
          {
            dealiiTriCell &&nb_i1 = i_cell->my_dealii_cell->neighbor(i_face);
            int face_nb_i1 = i_cell->my_dealii_cell->neighbor_face_no(i_face);
            std::stringstream nb_ss_id;
            nb_ss_id << nb_i1->id();
            std::string nb_str_id = nb_ss_id.str();
            if (nb_i1->subdomain_id() == comm_rank)
            {
              auto nb_manager =
                in_model->template get_owned_cell_manager<CellManagerType>(
                  nb_str_id);
              i_manager->set_cell_properties(i_face, comm_rank, 0);
              i_manager->set_owned_unkn_ids(i_face, i_local_unkn_on_this_rank,
                                            i_global_unkn_on_this_rank,
                                            n_unkns_per_dofs);
              nb_manager->set_cell_properties(face_nb_i1, comm_rank, 0);
              nb_manager->set_owned_unkn_ids(
                face_nb_i1, i_local_unkn_on_this_rank,
                i_global_unkn_on_this_rank, n_unkns_per_dofs);
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
                auto nb_manager =
                  in_model->template get_ghost_cell_manager<CellManagerType>(
                    nb_str_id);
                nb_manager->set_cell_properties(face_nb_i1, comm_rank, 0);
                nb_manager->set_owned_unkn_ids(
                  face_nb_i1, i_local_unkn_on_this_rank,
                  i_global_unkn_on_this_rank, n_unkns_per_dofs);
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
                i_global_unkn_on_this_rank += n_open_unkns_on_this_face;
                ++mpi_request_counter;
              }
              else
              {
                i_manager->set_cell_properties(i_face, nb_i1->subdomain_id(),
                                               0);
                i_manager->set_local_unkn_ids(i_face, i_local_unkn_on_this_rank,
                                              n_unkns_per_dofs);
                face_to_rank_recver[nb_i1->subdomain_id()]++;
                if (!is_there_a_msg_from_rank[nb_i1->subdomain_id()])
                  is_there_a_msg_from_rank[nb_i1->subdomain_id()] = true;
                ++mpi_status_counter;
              }
            }
            i_local_unkn_on_this_rank += n_open_unkns_on_this_face;
          }
        }
      }
    }
  }

  //
  // The next two variables contain num faces from rank zero to the
  // current rank, including and excluding current rank
  //
  std::vector<unsigned> unkns_count_be4_rank(comm_size, 0);
  std::vector<unsigned> unkns_count_up2_rank(comm_size, 0);
  unsigned n_unkns_this_rank_owns = i_global_unkn_on_this_rank;

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
      unkns_count_be4_rank[i_num] += unkns_count_up2_rank[j_num];
    }

  for (std::unique_ptr<cell<dim> > &i_cell : in_model->all_owned_cells)
  {
    auto i_manager = static_cast<ModelEq *>(i_cell.get())
                       ->template get_manager<CellManagerType>();
    i_manager->offset_global_unkn_ids(unkns_count_be4_rank[comm_rank]);
  }

  for (std::unique_ptr<cell<dim> > &ghost_cell : in_model->all_ghost_cells)
  {
    auto ghost_manager = static_cast<ModelEq *>(ghost_cell.get())
                           ->template get_manager<CellManagerType>();
    ghost_manager->offset_global_unkn_ids(unkns_count_be4_rank[comm_rank]);
  }

  //
  // Now, we want to also assign a unique number to those faces which
  // do not belong to the current rank. This includes those faces which
  // are connected to a locally owned cell or not. We start ghost face
  // numbers from -10, and go down. In any case we do not count faces
  // with closed degrees of freedom.
  //
  int ghost_unkns_counter = -10;
  for (std::unique_ptr<cell<dim> > &ghost_cell : in_model->all_ghost_cells)
  {
    auto ghost_manager = static_cast<ModelEq *>(ghost_cell.get())
                           ->template get_manager<CellManagerType>();
    auto i_basis =
      static_cast<ModelEq *>(ghost_cell.get())->template get_basis<BasisType>();
    std::vector<unsigned> n_unkns_per_dofs = i_basis->get_n_unkns_per_dofs();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      unsigned n_open_unkns_on_this_face =
        ghost_manager->get_n_open_unknowns_on_face(i_face, n_unkns_per_dofs);
      if (ghost_manager->face_is_not_visited(i_face))
      {
        const auto &face_i1 = ghost_cell->my_dealii_cell->face(i_face);
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
              i_face, ghost_cell->my_dealii_cell->subdomain_id(), 0);
            ghost_manager->set_ghost_unkn_ids(i_face, ghost_unkns_counter,
                                              n_unkns_per_dofs);
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
            i_face, ghost_cell->my_dealii_cell->subdomain_id(), 0);
          ghost_manager->set_ghost_unkn_ids(i_face, ghost_unkns_counter,
                                            n_unkns_per_dofs);
          if (face_i1->has_children())
          {
            int face_nb_subface =
              ghost_cell->my_dealii_cell->neighbor_face_no(i_face);
            for (unsigned i_subface = 0;
                 i_subface < face_i1->number_of_children();
                 ++i_subface)
            {
              dealiiTriCell &&nb_subface =
                ghost_cell->my_dealii_cell->neighbor_child_on_subface(
                  i_face, i_subface);
              if (nb_subface->is_ghost())
              {
                auto nb_manager =
                  in_model->template get_ghost_cell_manager<CellManagerType>(
                    nb_subface);
                nb_manager->set_cell_properties(
                  face_nb_subface, nb_subface->subdomain_id(), i_subface + 1);
                nb_manager->set_ghost_unkn_ids(
                  face_nb_subface, ghost_unkns_counter, n_unkns_per_dofs);
              }
            }
          }
          else if (ghost_cell->my_dealii_cell->neighbor(i_face)->is_ghost())
          {
            dealiiTriCell &&nb_i1 =
              ghost_cell->my_dealii_cell->neighbor(i_face);
            int face_nb_i1 =
              ghost_cell->my_dealii_cell->neighbor_face_no(i_face);
            auto nb_manager =
              in_model->template get_ghost_cell_manager<CellManagerType>(nb_i1);
            assert(nb_manager->unkns_id_in_this_rank[face_nb_i1].size() == 0);
            assert(nb_manager->unkns_id_in_all_ranks[face_nb_i1].size() == 0);
            nb_manager->set_cell_properties(face_nb_i1, nb_i1->subdomain_id(),
                                            0);
            nb_manager->set_ghost_unkn_ids(face_nb_i1, ghost_unkns_counter,
                                           n_unkns_per_dofs);
          }
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
                  in_model->my_mesh->refn_cycle,
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
                   in_model->my_mesh->refn_cycle,
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
                 in_model->my_mesh->refn_cycle,
                 *my_comm,
                 &all_mpi_stats_of_rank[recv_counter]);
        std::vector<std::string> tokens;
        Tokenize(buffer, tokens, "#");
        assert(tokens.size() == 4);
        std::string cell_unique_id = tokens[0];
        unsigned face_num = std::stoi(tokens[1]);
        auto i_manager =
          in_model->template get_owned_cell_manager<CellManagerType>(
            cell_unique_id);
        auto i_basis =
          in_model->template get_owned_cell_basis<BasisType>(cell_unique_id);
        std::vector<unsigned> n_unkns_per_dofs =
          i_basis->get_n_unkns_per_dofs();
        assert(i_manager->dof_status_on_faces[face_num].count() != 0);
        assert(i_manager->unkns_id_in_all_ranks[face_num].size() == 0);
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
                  in_model->my_mesh->refn_cycle,
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
                   in_model->my_mesh->refn_cycle,
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
                 in_model->my_mesh->refn_cycle,
                 *my_comm,
                 &all_mpi_stats_of_rank[recv_counter]);
        std::vector<std::string> tokens;
        Tokenize(buffer, tokens, "#");
        assert(tokens.size() == 4);
        std::string cell_unique_id = tokens[0];
        unsigned face_num = std::stoi(tokens[1]);
        auto i_manager =
          in_model->template get_owned_cell_manager<CellManagerType>(
            cell_unique_id);
        auto i_basis =
          in_model->template get_owned_cell_basis<BasisType>(cell_unique_id);
        std::vector<unsigned> n_unkns_per_dofs =
          i_basis->get_n_unkns_per_dofs();
        assert(i_manager->dof_status_on_faces[face_num].count() != 0);
        assert(i_manager->unkns_id_in_all_ranks[face_num].size() == 0);
        //
        // The DOF data received from other CPU is the ID of the first
        // DOF of this face.
        //
        i_manager->set_nonlocal_unkn_ids(
          face_num, std::stoi(tokens[3]) + unkns_count_be4_rank[i_recv->first],
          n_unkns_per_dofs);
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

  using CellIterType = typename cell<dim, spacedim>::CellIterType;
  struct unkn_properties
  {
    unkn_properties()
      : n_local_connected_unkns(0), n_nonlocal_connected_unkns(0)
    {
    }
    unsigned n_local_connected_unkns;
    unsigned n_nonlocal_connected_unkns;
    std::vector<CellIterType> parent_cells;
    std::vector<unsigned> connected_face_of_parent_cell;
    std::vector<CellIterType> parent_ghosts;
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
  // unkn which belongs to this rank (So those unkns which do
  // not belong to this rank are not present in this vector !).
  // Then, fill the parent_cells vector with
  // those parent cells which also belong to the current rank.
  // Also, we fill parent_ghosts with ghost cells
  // connected to the current face.
  //
  std::vector<unkn_properties> all_owned_unkns(n_unkns_this_rank_owns);
  for (CellIterType cell_it = in_model->all_owned_cells.begin();
       cell_it != in_model->all_owned_cells.end();
       ++cell_it)
  {
    auto i_manager = static_cast<ModelEq *>(cell_it->get())
                       ->template get_manager<CellManagerType>();
    auto i_basis =
      static_cast<ModelEq *>(cell_it->get())->template get_basis<BasisType>();
    std::vector<unsigned> n_unkns_per_dofs = i_basis->get_n_unkns_per_dofs();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      unsigned n_open_unkns_on_this_face =
        i_manager->get_n_open_unknowns_on_face(i_face, n_unkns_per_dofs);
      if (i_manager->face_owner_rank[i_face] == comm_rank)
      {
        for (unsigned i_unkn = 0;
             i_unkn < i_manager->unkns_id_in_all_ranks[i_face].size();
             ++i_unkn)
        {
          if (i_manager->unkns_id_in_all_ranks[i_face][i_unkn] != -1)
          {
            assert(i_manager->unkns_id_in_all_ranks[i_face][i_unkn] >= 0);
            int unkn_i1 = i_manager->unkns_id_in_all_ranks[i_face][i_unkn] -
                          unkns_count_be4_rank[comm_rank];
            all_owned_unkns[unkn_i1].parent_cells.push_back(cell_it);
            all_owned_unkns[unkn_i1].connected_face_of_parent_cell.push_back(
              i_face);
            //
            // Here, we just add the open DoFs on the face itself. The reason
            // for this is to avoid adding the connected dofs of each face
            // to itself two times. This is the reason that we check if the
            // n_local_connected_unkns are zero.
            //
            if (all_owned_unkns[unkn_i1].n_local_connected_unkns == 0)
              all_owned_unkns[unkn_i1].n_local_connected_unkns =
                n_open_unkns_on_this_face;
          }
        }
      }
    }
  }

  for (CellIterType ghost_cell_it = in_model->all_ghost_cells.begin();
       ghost_cell_it != in_model->all_ghost_cells.end();
       ++ghost_cell_it)
  {
    auto ghost_manager = static_cast<ModelEq *>(ghost_cell_it->get())
                           ->template get_manager<CellManagerType>();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      if (ghost_manager->face_owner_rank[i_face] == comm_rank)
      {
        for (unsigned i_unkn = 0;
             i_unkn < ghost_manager->unkns_id_in_all_ranks[i_face].size();
             ++i_unkn)
        {
          if (ghost_manager->unkns_id_in_all_ranks[i_face][i_unkn] != -1)
          {
            assert(ghost_manager->unkns_id_in_all_ranks[i_face][i_unkn] >= 0);
            int unkn_i1 = ghost_manager->unkns_id_in_all_ranks[i_face][i_unkn] -
                          unkns_count_be4_rank[comm_rank];
            all_owned_unkns[unkn_i1].parent_ghosts.push_back(ghost_cell_it);
            all_owned_unkns[unkn_i1].connected_face_of_parent_ghost.push_back(
              i_face);
            //
            // Here we assert that the following condition is never satiisfied.
            //
            if (all_owned_unkns[unkn_i1].n_local_connected_unkns == 0)
            {
              std::cout << "This is a curious case which should not happen. "
                           "How is that a ghost cell can give a face ownership?"
                        << std::endl;
              assert(all_owned_unkns[unkn_i1].n_local_connected_unkns != 0);
            }
          }
        }
      }
    }
  }

  this->n_global_unkns_rank_owns = n_unkns_this_rank_owns;

  //
  //
  //
  for (unkn_properties &unkn : all_owned_unkns)
  {
    std::map<int, unsigned> local_unkns_num_map;
    std::map<int, unsigned> nonlocal_unkns_num_map;

    for (unsigned i_parent_cell = 0; i_parent_cell < unkn.parent_cells.size();
         ++i_parent_cell)
    {
      auto parent_cell = unkn.parent_cells[i_parent_cell];
      auto parent_manager = static_cast<ModelEq *>(parent_cell->get())
                              ->template get_manager<CellManagerType>();
      for (unsigned j_face = 0; j_face < 2 * dim; ++j_face)
      {
        unsigned face_ij = unkn.connected_face_of_parent_cell[i_parent_cell];
        if (j_face != face_ij)
        {
          for (unsigned i_unkn = 0;
               i_unkn < parent_manager->unkns_id_in_all_ranks[j_face].size();
               ++i_unkn)
          {
            if (parent_manager->face_owner_rank[j_face] == comm_rank &&
                parent_manager->unkns_id_in_all_ranks[j_face][i_unkn] != -1)
              local_unkns_num_map[parent_manager
                                    ->unkns_id_in_all_ranks[j_face][i_unkn]]++;
            if (parent_manager->face_owner_rank[j_face] != comm_rank &&
                parent_manager->unkns_id_in_all_ranks[j_face][i_unkn] != -1)
              nonlocal_unkns_num_map
                [parent_manager->unkns_id_in_all_ranks[j_face][i_unkn]]++;
          }
        }
      }
    }

    for (unsigned i_parent_ghost = 0;
         i_parent_ghost < unkn.parent_ghosts.size();
         ++i_parent_ghost)
    {
      auto parent_ghost = unkn.parent_ghosts[i_parent_ghost];
      auto ghost_manager = static_cast<ModelEq *>(parent_ghost->get())
                             ->template get_manager<CellManagerType>();
      for (unsigned j_face = 0; j_face < 2 * dim; ++j_face)
      {
        unsigned face_ij = unkn.connected_face_of_parent_ghost[i_parent_ghost];
        if (j_face != face_ij)
          for (unsigned i_unkn = 0;
               i_unkn < ghost_manager->unkns_id_in_all_ranks[j_face].size();
               ++i_unkn)
          {
            if (ghost_manager->face_owner_rank[j_face] == comm_rank &&
                ghost_manager->unkns_id_in_all_ranks[j_face][i_unkn] != -1)
              local_unkns_num_map[ghost_manager
                                    ->unkns_id_in_all_ranks[j_face][i_unkn]]++;
            if (ghost_manager->face_owner_rank[j_face] != comm_rank &&
                ghost_manager->unkns_id_in_all_ranks[j_face][i_unkn] != -1)
              nonlocal_unkns_num_map
                [ghost_manager->unkns_id_in_all_ranks[j_face][i_unkn]]++;
          }
      }
    }
    unkn.n_local_connected_unkns += local_unkns_num_map.size();
    unkn.n_nonlocal_connected_unkns = nonlocal_unkns_num_map.size();
  }

  MPI_Allreduce(&this->n_global_unkns_rank_owns,
                &this->n_global_unkns_on_all_ranks,
                1,
                MPI_UNSIGNED,
                MPI_SUM,
                *my_comm);

  unsigned unkn_counter = 0;
  this->n_local_unkns_connected_to_unkn.resize(this->n_global_unkns_rank_owns);
  this->n_nonlocal_unkns_connected_to_unkn.resize(
    this->n_global_unkns_rank_owns);
  for (unkn_properties &i_unkn : all_owned_unkns)
  {
    this->n_local_unkns_connected_to_unkn[unkn_counter] +=
      i_unkn.n_local_connected_unkns;
    this->n_nonlocal_unkns_connected_to_unkn[unkn_counter] +=
      i_unkn.n_nonlocal_connected_unkns;
    ++unkn_counter;
  }

  unsigned some_shit = 0;

  std::map<unsigned, unsigned> map_from_local_to_global;
  for (std::unique_ptr<cell<dim, spacedim> > &i_cell :
       in_model->all_owned_cells)
  {
    auto i_manager = static_cast<ModelEq *>(i_cell.get())
                       ->template get_manager<CellManagerType>();
    for (unsigned i_face = 0; i_face < 2 * dim; ++i_face)
    {
      for (unsigned i_unkn = 0;
           i_unkn < i_manager->unkns_id_in_this_rank[i_face].size();
           ++i_unkn)
      {
        int index1 = i_manager->unkns_id_in_this_rank[i_face][i_unkn];
        int index2 = i_manager->unkns_id_in_all_ranks[i_face][i_unkn];

        //        printf("%d %d %d %d %d %d\n", comm_rank, some_shit, i_face,
        //        i_unkn,
        //               index1, index2);

        if (index1 != -1)
        {
          assert(index1 >= 0 && index2 >= 0);
          map_from_local_to_global[index1] = index2;
        }
      }
    }

    ++some_shit;
  }
  assert(map_from_local_to_global.size() == i_local_unkn_on_this_rank);

  this->n_local_unkns_on_this_rank = i_local_unkn_on_this_rank;
  this->scatter_from.reserve(this->n_local_unkns_on_this_rank);
  this->scatter_to.reserve(this->n_local_unkns_on_this_rank);
  for (const auto &map_it : map_from_local_to_global)
  {
    this->scatter_to.push_back(map_it.first);
    this->scatter_from.push_back(map_it.second);
  }

  /*
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
