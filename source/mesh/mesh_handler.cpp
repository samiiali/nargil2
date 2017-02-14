#include "../../include/mesh/mesh_handler.hpp"

template <int dim, int spacedim> void nargil::mesh<dim, spacedim>::write_grid()
{
  int comm_rank, comm_size, refn_cycle;
  MPI_Comm_rank(*my_comm, &comm_rank);
  MPI_Comm_size(*my_comm, &comm_size);
  refn_cycle = 0;
  dealii::GridOut Grid1_Out;
  dealii::GridOutFlags::Svg svg_flags(
    1,                                       // line_thickness = 2,
    2,                                       // boundary_line_thickness = 4,
    false,                                   // margin = true,
    dealii::GridOutFlags::Svg::transparent,  // background = white,
    0,                                       // azimuth_angle = 0,
    0,                                       // polar_angle = 0,
    dealii::GridOutFlags::Svg::subdomain_id, // coloring = level_number,
    false, // convert_level_number_to_height = false,
    false, // label_level_number = true,
    true,  // label_cell_index = true,
    false, // label_material_id = false,
    false, // label_subdomain_id = false,
    true,  // draw_colorbar = true,
    true); // draw_legend = true
  Grid1_Out.set_flags(svg_flags);
  if (dim == 2)
  {
    std::ofstream Grid1_OutFile("the_grid" + std::to_string(refn_cycle) +
                                std::to_string(comm_rank) + ".svg");
    Grid1_Out.write_svg(tria, Grid1_OutFile);
  }
  else
  {
    std::ofstream Grid1_OutFile("the_grid" + std::to_string(refn_cycle) +
                                std::to_string(comm_rank) + ".msh");
    Grid1_Out.write_msh(tria, Grid1_OutFile);
  }
}

template <int dim, int spacedim>
nargil::mesh<dim, spacedim>::mesh(const MPI_Comm &comm_,
                                  const unsigned n_threads_,
                                  const bool adaptive_on_)
  : my_comm(&comm_),
    adaptive_on(adaptive_on_),
    n_threads(n_threads_),
    tria(*my_comm),
    refn_cycle(0)
{
  //  int comm_rank, comm_size;
  //  MPI_Comm_rank(*comm, &comm_rank);
  //  MPI_Comm_size(*comm, &comm_size);
}

template <int dim, int spacedim> nargil::mesh<dim, spacedim>::~mesh() {}

template <int dim, int spacedim>
void nargil::mesh<dim, spacedim>::init_cell_ID_to_num()
{
  unsigned n_cell = 0;
  n_owned_cell = 0;
  n_ghost_cell = 0;
  for (dealiiTriCell &&i_cell : tria.active_cell_iterators())
  {
    if (i_cell->is_locally_owned())
    {
      std::stringstream ss_id;
      ss_id << i_cell->id();
      std::string cell_id = ss_id.str();
      owned_cell_ID_to_num[cell_id] = n_owned_cell;
      ++n_owned_cell;
    }
    if (i_cell->is_ghost())
    {
      std::stringstream ss_id;
      ss_id << i_cell->id();
      std::string cell_id = ss_id.str();
      ghost_cell_ID_to_num[cell_id] = n_ghost_cell;
      ++n_ghost_cell;
    }
    ++n_cell;
  }
}

//
//

template <int dim, int spacedim>
int nargil::mesh<dim, spacedim>::cell_id_to_num_finder(
  const dealiiTriCell &in_dealii_cell, const bool cell_is_owned) const
{
  if (cell_is_owned)
  {
    std::stringstream cell_id;
    cell_id << in_dealii_cell->id();
    std::string cell_str_id = cell_id.str();
    auto it_1 = owned_cell_ID_to_num.find(cell_str_id);
    try
    {
      if (it_1 != owned_cell_ID_to_num.end())
        return it_1->second;
      else
        throw 101;
    }
    catch (int)
    {
      std::cout << "The owned cell is not found in mesh. Contact Ali."
                << std::endl;
    }
  }
  else
  {
    std::stringstream cell_id;
    cell_id << in_dealii_cell->id();
    std::string cell_str_id = cell_id.str();
    auto it_1 = ghost_cell_ID_to_num.find(cell_str_id);
    try
    {
      if (it_1 != ghost_cell_ID_to_num.end())
        return it_1->second;
      else
        throw 102;
    }
    catch (int)
    {
      std::cout << "The ghost cell is not found in mesh. Contact Ali."
                << std::endl;
    }
  }
  return -1;
}

//
//

template <int dim, int spacedim>
int nargil::mesh<dim, spacedim>::cell_id_to_num_finder(
  const std::string &cell_str_id, const bool cell_is_owned) const
{
  if (cell_is_owned)
  {
    auto it_1 = owned_cell_ID_to_num.find(cell_str_id);
    try
    {
      if (it_1 != owned_cell_ID_to_num.end())
        return it_1->second;
      else
        throw 101;
    }
    catch (int)
    {
      std::cout << "The owned cell is not found in mesh. Contact Ali."
                << std::endl;
    }
  }
  else
  {
    auto it_1 = ghost_cell_ID_to_num.find(cell_str_id);
    try
    {
      if (it_1 != ghost_cell_ID_to_num.end())
        return it_1->second;
      else
        throw 102;
    }
    catch (int)
    {
      std::cout << "The ghost cell is not found in mesh. Contact Ali."
                << std::endl;
    }
  }
  return -1;
}

//
//

template <int dim, int spacedim>
template <typename F>
void nargil::mesh<dim, spacedim>::generate_mesh(F mesh_gen_func)
{
  mesh_gen_func(tria);
  init_cell_ID_to_num();
  write_grid();
}
