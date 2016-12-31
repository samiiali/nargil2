#include "../../include/mesh/mesh_handler.hpp"

template <int dim, int spacedim> void nargil::mesh<dim, spacedim>::write_grid()
{
  int comm_rank, comm_size, refn_cycle;
  MPI_Comm_rank(*comm, &comm_rank);
  MPI_Comm_size(*comm, &comm_size);
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
  : comm(&comm_),
    adaptive_on(adaptive_on_),
    n_threads(n_threads_),
    tria(*comm,
         typename dealii::Triangulation<dim>::MeshSmoothing(
           dealii::Triangulation<dim>::smoothing_on_refinement |
           dealii::Triangulation<dim>::smoothing_on_coarsening))
{
  int comm_rank, comm_size;
  MPI_Comm_rank(*comm, &comm_rank);
  MPI_Comm_size(*comm, &comm_size);
}

template <int dim, int spacedim> nargil::mesh<dim, spacedim>::~mesh() {}

template <int dim, int spacedim>
void nargil::mesh<dim, spacedim>::init_cell_ID_to_num()
{
  unsigned n_cell = 0;
  n_owned_cell = 0;
  n_ghost_cell = 0;
  for (dealii_cell_type &&cell : tria.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      std::stringstream ss_id;
      ss_id << cell->id();
      std::string cell_id = ss_id.str();
      cell_ID_to_num[cell_id] = n_owned_cell;
      ++n_owned_cell;
    }
    if (cell->is_ghost())
    {
      ++n_ghost_cell;
    }
    ++n_cell;
  }
}

template <int dim, int spacedim>
template <typename F>
void nargil::mesh<dim, spacedim>::generate_mesh(F generate_mesh_)
{
  generate_mesh_(tria);
  init_cell_ID_to_num();
  write_grid();
}
