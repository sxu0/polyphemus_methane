// Copyright (C) 2015, ENPC
//    Author(s): Eve Lecoeure, Yelva Roustan, Carole Legorgeu
//
// This file is part of the air quality modeling system Polyphemus.
//
// Polyphemus is developed in the INRIA - ENPC joint project-team CLIME and in
// the ENPC - EDF R&D joint laboratory CEREA.
//
// Polyphemus is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
//
// Polyphemus is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
// more details.
//
// For more information, visit the Polyphemus web site:
//      http://cerea.enpc.fr/polyphemus/
//
// >>>> Made for MOZART4/geos5 input <<<<


//////////////
// INCLUDES //
//////////////

#include <cmath>
#include <iostream>
#include <vector>
#include <sstream>
#include <map>
using namespace std;

#define SELDONDATA_DEBUG_LEVEL_2
#define SELDONDATA_WITH_NETCDF

#include "Common.cxx"
using namespace Polyphemus;

#include "SeldonData.hxx"
using namespace SeldonData;

#include "AtmoData.hxx"
using namespace AtmoData;


// List of species associated with a ratio.
class SpeciesInfo
{
public:
  vector<string> species_in;
  vector<float> ratio;
};

int main(int argc, char** argv)
{

  TRY;
  cout << endl;
  string main_config_file("general.cfg"), sec_config_file("mozart4.cfg");

  if (argc != 3 && argc != 4 && argc != 5)
    {
      string mesg  = "Usage:\n";
      mesg += string("  ") + argv[0] + " [main configuration file] [secondary configuration file] [First date] [Second date]\n";
      mesg += string("  ") + argv[0] + " [secondary config file] [First date] [Second date]\n";
      mesg += string("  ") + argv[0] + " [First date] [Second date]\n";
      mesg += "Arguments:\n";
      mesg += "  [main configuration file] (optional): main configuration file. Default= general.cfg\n";
      mesg += "  [secondary configuration file] (optional): secondary configuration file. Default= mozart4.cfg\n";
      mesg += "  [First date]: first date of the simulation.\n";
      mesg += "  [Second date]: end date of the simulation\n";
      cout << mesg << endl;
      return 1;
    } // end if (argc != 3 && argc != 4 && argc != 5)

  if (argc > 3) sec_config_file = argv[2];
  if (argc > 4) main_config_file = argv[1];

  string date_beg_str = argv[argc - 2];
  string date_end_str = argv[argc - 1];

  if (!exists(main_config_file))
    throw "Unable to find configuration file \"" + main_config_file + "\".";

  cout << "==> GET BOUNDARY CONDITIONS FOR GAS <===" << endl;
  cout << endl;

  ////////////////////////
  // FIRST DECLARATIONS //
  ////////////////////////

  typedef float real;

  // Temporary values.
  int h, i, j, k, l;

  // Configuration.
  ConfigStreams config(main_config_file);
  if (exists(sec_config_file))
    config.AddFile(sec_config_file);

  // Constants.
  const real Rgas = 286.9969;
  const real mw_air = 29.0;

  Date date_beg(date_beg_str);
  Date date_end(date_end_str);

  cout << "Get MOZART4-GEOS5 data from " << date_beg.GetDate("%y-%m-%d_%h");
  cout << " to " << date_end.GetDate("%y-%m-%d_%h") << endl;

  ////////////////////////
  // READ CONFIGURATION //
  ////////////////////////

  cout << "Read configuration... " ;
  cout.flush();

  // Output domain.
  int Nz_out, Ny_out, Nx_out;
  real Delta_y_out, Delta_x_out, y_min_out, x_min_out;
  string vertical_levels;

  config.SetSection("[domain]");
  config.PeekValue("Nz", "> 0", Nz_out);
  config.PeekValue("Ny", "> 0", Ny_out);
  config.PeekValue("Nx", "> 0", Nx_out);
  config.PeekValue("Delta_y", "> 0", Delta_y_out);
  config.PeekValue("Delta_x", "> 0", Delta_x_out);
  config.PeekValue("y_min", y_min_out);
  config.PeekValue("x_min", x_min_out);
  config.PeekValue("Vertical_levels", vertical_levels);

  // Input/output directories and files.
  string Directory_out, mozart_file;
  string species_file, molecular_weights;

  config.SetSection("[input_files]");
  config.PeekValue("Directory_bc", Directory_out);
  config.PeekValue("File_mozart", mozart_file);
  config.PeekValue("Species_GAS", species_file);
  config.PeekValue("Molecular_weights_GAS", molecular_weights);

  Directory_out += "/";

  ///////////////////////////
  // MOZART MAP OF SPECIES //
  ///////////////////////////

  cout << "Get speciation from input to output species... ";
  cout.flush();

  // File "species.txt" format (one line):
  // [Mozart species] [Polair species] {[ratio] {[Polair species] [ratio] {..}}}
  ifstream species(species_file.c_str());

  if (!species.is_open())
    throw string("Unable to open file \"") + species_file + "\".";

  string line, species_in, species_out;

  string delim = " \t\n|";

  // To all output (Polair) species, we associate a list of input (Mozart)
  // species and their ratios (i.e. the proportion of the input species to
  // be put in the output species).
  map<string, SpeciesInfo> species_map;
  map<string, SpeciesInfo>::iterator pos;

  // For all lines.
  while (getline(species, line))
    {
      // 'species_data' will be filled with all words on the line.
      vector<string> species_data;

      string::size_type beg_index, end_index;
      beg_index = line.find_first_not_of(delim);

      // For each "word".
      while (beg_index != string::npos)
        {
          end_index = line.find_first_of(delim, beg_index);
          if (end_index == string::npos)
            end_index = line.length();

          // Put the word in 'species_data'.
          species_data.push_back(line.substr(beg_index, end_index - beg_index));

          beg_index = line.find_first_not_of(delim, end_index);
        }

      // Number of words on the line (Mozart species and Polair species and ratios).
      int size = species_data.size();

      // If the input species is associated with any output species.
      if (size > 1)
        {
          // Number of output species.
          int nb = max((size - 1) / 2, 1);
          // For all output species, update its list of input species and ratios.
          for (i = 0; i < nb; i++)
            {
              // If the output species is not in the map yet.
              if ((pos = species_map.find(species_data[2 * i + 1])) == species_map.end())
                {
                  SpeciesInfo info;
                  info.species_in.push_back(species_data[0]);
                  if (size % 2 == 1) // If ratios are specified.
                    info.ratio.push_back(to_num<float>(species_data[2 * (i + 1)]));
                  else  // No ratio: assumed to be 1.
                    info.ratio.push_back(1.);
                  species_map.insert(make_pair(species_data[2 * i + 1], info));
                }
              else  // the output is in the map.
                {
                  pos->second.species_in.push_back(species_data[0]);
                  if (size % 2 == 1) // If ratios are specified.
                    pos->second.ratio.push_back(to_num<float>(species_data[2 * (i + 1)]));
                  else  // No ratio: assumed to be 1.
                    pos->second.ratio.push_back(1.);
                }
            }
        }
    }  // Loop over 'species_file' lines.
  species.close();

  ////////////////////////////
  // OUTPUT SPECIES WEIGHTS //
  ////////////////////////////

  cout << "Get molecular weights of input species... ";
  cout.flush();

  map<string, float> weights_map;
  map<string, float>::iterator weights_pos;

  ifstream weights(molecular_weights.c_str());

  if (!weights.is_open())
    throw string("Unable to open file \"") + molecular_weights + "\".";

  float weight;

  // For all lines.
  while (getline(weights, line))
    {
      istringstream sline(line);
      sline >> species_out;
      // If there is still species.
      if (sline.good())
        {
          sline >> weight;
          weights_map.insert(make_pair(species_out, weight));
        } // end if (sline.good())
    } // end while (getline(weights, line))
  weights.close();

  //////////////////
  // MOZART GRIDS //
  //////////////////

  // Input settings
  cout << "Define MOZART4 domain... " ;
  cout.flush();

  // Input domain.
  int Nt_in, Nz_in, Nz_in_tmp, Ny_in, Nx_in;

  FormatNetCDF<float> Mozart;
  Mozart.ReadDimension(mozart_file, "date", 0, Nt_in);
  Mozart.ReadDimension(mozart_file, "lev", 0, Nz_in_tmp);
  Mozart.ReadDimension(mozart_file, "lat", 0, Ny_in);
  Mozart.ReadDimension(mozart_file, "lon", 0, Nx_in);

  Nz_in = Nz_in_tmp + 1;

  // Input grids.
  RegularGrid<real> GridT_in(Nt_in);
  RegularGrid<double> GridT_in_double(Nt_in);
  RegularGrid<int> GridT_in_days(Nt_in);
  RegularGrid<int> GridT_in_seconds(Nt_in);

  RegularGrid<real> GridY_in(Ny_in);
  RegularGrid<real> GridX_in(Nx_in);

  // Reads MOZART4
  Mozart.Read(mozart_file, "time", GridT_in_double);
  Mozart.Read(mozart_file, "date", GridT_in_days);
  Mozart.Read(mozart_file, "datesec", GridT_in_seconds);

  Mozart.Read(mozart_file, "lat", GridY_in);
  Mozart.Read(mozart_file, "lon", GridX_in);

  // Conversion lon from <0 to 360> to <-180 to 180>
  for (i = 0; i < Nx_in; i++)
    if (GridX_in(i) >= 180 && GridX_in(i) <= 360)
      GridX_in(i) = GridX_in(i) - 360;

  // Sets time steps.
  for (i = 0; i < Nt_in; i++)
    GridT_in(i) = GridT_in_double(i);

  //Find correspondance for dates
  string index_date_tmp;
  int i1 = -1, i2 = -1;

  for (h = 0; h < Nt_in; h++)
    {
      if (GridT_in_seconds(h) < 36000)
        index_date_tmp = to_str(GridT_in_days(h)) + "0" + to_str((GridT_in_seconds(h) / 3600));
      else
        index_date_tmp = to_str(GridT_in_days(h)) + to_str((GridT_in_seconds(h) / 3600));
      if (index_date_tmp == to_str(date_beg.GetDate("%y%m%d%h")))
        i1 = h;
      else if (index_date_tmp == to_str(date_end.GetDate("%y%m%d%h")))
        i2 = h;
    }

  if (i1 == -1)
    throw string("\n\n Unable to find data for ") + to_str(date_beg.GetDate("%y%m%d%h"));
  if (i2 == -1)
    throw string("\n\n Unable to find data for ") + to_str(date_end.GetDate("%y%m%d%h"));

  vector<int> Indexh; // list time index
  for (h = i1; h <= i2; h++)
    Indexh.push_back((int)h);

  int Nt_out = Indexh.size();

  //Output settings
  cout << "Define output domain... " ;
  cout.flush();

  FormatBinary<float> Polair;

  // Output grids.
  RegularGrid<real> GridT_out(Nt_out);
  RegularGrid<real> GridZ_out(Nz_out);
  RegularGrid<real> GridY_out(y_min_out, Delta_y_out, Ny_out);
  RegularGrid<real> GridX_out(x_min_out, Delta_x_out, Nx_out);

  // For boundary conditions.  All interfaces along z.
  RegularGrid<real> GridZ_all_interf_out(Nz_out + 1);

  // Boundary layers along z, y and x direction.
  RegularGrid<real> GridZ_interf_out(1);
  RegularGrid<real> GridY_interf_out(y_min_out - Delta_y_out / 2., (Ny_out + 1) * Delta_y_out, 2);
  RegularGrid<real> GridX_interf_out(x_min_out - Delta_x_out / 2., (Nx_out + 1) * Delta_x_out, 2);

  // Sets time steps.
  for (i = 0; i < Nt_out; i++)
    GridT_out(i) = GridT_in_double(Indexh[i]);

  // Reads output altitudes.
  FormatText Heights_out;
  Heights_out.Read(vertical_levels, GridZ_all_interf_out);

  // Sets values at nodes.
  for (k = 0; k < Nz_out; k++)
    GridZ_out(k) = (GridZ_all_interf_out(k) + GridZ_all_interf_out(k + 1)) / 2.;

  // Sets the boundary layer altitude.
  GridZ_interf_out(0) = 1.5 * GridZ_all_interf_out(Nz_out - 1) - 0.5 * GridZ_all_interf_out(Nz_out - 2);

  cout << "done" << endl;

  //////////
  // Data //
  //////////

  // Input fields.
  // suffix '_in_tmp' = initial Mozart grid.

  cout << "Memory allocation for input fields... " ;
  cout.flush();

  GeneralGrid<real, 4> GridZ_in_tmp(shape(Nt_in, Nz_in_tmp, Ny_in, Nx_in), 1, shape(0, 1, 2, 3));
  GeneralGrid<real, 4> GridZ_in(shape(Nt_in, Nz_in, Ny_in, Nx_in), 1, shape(0, 1, 2, 3));
  GridZ_in_tmp.SetVariable(1);
  GridZ_in_tmp.SetDuplicate(false);
  GridZ_in.SetVariable(1);
  GridZ_in.SetDuplicate(false);

  Data<real, 4> Temperature_tmp(GridT_in, GridZ_in_tmp, GridY_in, GridX_in);
  Data<real, 1> alpha(Nz_in_tmp), beta(Nz_in_tmp);
  Data<real, 3> SurfacePressure(GridT_in, GridY_in, GridX_in);
  Data<real, 4> Pressure_tmp(GridT_in, GridZ_in_tmp, GridY_in, GridX_in);
  Data<real, 4> Conc_in_tmp(GridT_in, GridZ_in_tmp, GridY_in, GridX_in);

  Data<real, 4> Temperature(GridT_in, GridZ_in, GridY_in, GridX_in);
  Data<real, 4> Pressure(GridT_in, GridZ_in, GridY_in, GridX_in);
  Data<real, 4> Conc_in(GridT_in, GridZ_in, GridY_in, GridX_in);

  // Output fields.
  cout << "for output fields... " ;
  cout.flush();
  Data<real, 4> Conc_out_x_tmp(GridT_out, GridZ_out, GridY_out, GridX_interf_out);
  Data<real, 4> Conc_out_y_tmp(GridT_out, GridZ_out, GridY_interf_out, GridX_out);
  Data<real, 4> Conc_out_z_tmp(GridT_out, GridZ_interf_out, GridY_out, GridX_out);
  Data<real, 4> Conc_out_x(GridT_out, GridZ_out, GridY_out, GridX_interf_out);
  Data<real, 4> Conc_out_y(GridT_out, GridZ_out, GridY_interf_out, GridX_out);
  Data<real, 4> Conc_out_z(GridT_out, GridZ_interf_out, GridY_out, GridX_out);

  cout << "done" << endl << endl;

  /////////////////
  // READS INPUT //
  /////////////////

  cout << "Extract Temperature ...";
  cout.flush();
  Mozart.Read(mozart_file, "T", Temperature_tmp);

  cout << " Compute level heights ...";
  cout.flush();
  // Hybrid coefficients.
  Mozart.Read(mozart_file, "hyam", alpha);
  Mozart.Read(mozart_file, "hybm", beta);
  Mozart.Read(mozart_file, "PS", SurfacePressure);
  ComputePressure(alpha, beta, SurfacePressure, Pressure_tmp, real(100000.));

  // Note: MOZART4 data is provided from the top level to the bottom level
  // (altitude). The interpolation function requires coordinates
  // to be sorted in increasing order. So, data must be reversed...

  Temperature_tmp.ReverseData(1);
  Pressure_tmp.ReverseData(1);

  ComputeHeight(SurfacePressure, Pressure_tmp, Temperature_tmp, GridZ_in_tmp);

  // Note: to limit extrapolations near surface,
  // we duplicate datas from the first mozart4 level.
  // This new level is considered as the first output level.

  cout << " Arrange data ...";
  cout.flush();
  for (h = 0; h < Nt_in; h++)
    for (j = 0; j < Ny_in; j++)
      for (i = 0; i < Nx_in; i++)
        {
          GridZ_in.Value(h, 0, j, i) = GridZ_out(0);
          Temperature(h, 0, j, i) = Temperature_tmp(h, 0, j, i);
          Pressure(h, 0, j, i) = Pressure_tmp(h, 0, j, i);
          for (k = 1; k < Nz_in; k++)
            {
              GridZ_in.Value(h, k, j, i) = GridZ_in_tmp.Value(h, k - 1, j, i);
              Temperature(h, k, j, i) = Temperature_tmp(h, k - 1, j, i);
              Pressure(h, k, j, i) = Pressure_tmp(h, k - 1, j, i);
            } // end k
        } // end i

  cout << " done" << endl;

  /////////////////////////////
  // BOUNDARY CONCENTRATIONS //
  /////////////////////////////

  int isp, ipol, s;
  cout << "Compute BC for gas species... " << endl;

  cout << "<< From MOZART4 : Nt = " << Nt_in << " Nz = " << Nz_in << " Ny = " << Ny_in << " Nx = " << Nx_in << endl;
  cout << ">> to POLAIR : Nt = " << Nt_out << " Nz = " << Nz_out << " Ny = " << Ny_out << " Nx = " << Nx_out << endl;

  // For all output species (in the map 'species_map').
  for (pos = species_map.begin(); pos != species_map.end(); pos++)
    {
      // pos->first: output species name.
      // pos->second: input species names and ratios.

      cout << "     " << pos->first  ;

      Conc_out_x.SetZero();
      Conc_out_y.SetZero();
      Conc_out_z.SetZero();

      cout << " ... compute with : " << endl;

      // For all input species.
      for (l = 0; l < int(pos->second.species_in.size()); l++)
        {
          cout << "     \\=> " << pos->second.species_in[l] << " (" << pos->second.ratio[l] << ") : ";
          cout.flush();
          // Reads input species concentrations.
          cout << "read";
          cout.flush();
          Mozart.Read(mozart_file, pos->second.species_in[l] + "_VMR_inst", Conc_in_tmp);
          Conc_in_tmp.ReverseData(1);

          // Arrange input species to limit extrapolations.
          cout << ", arrange";
          cout.flush();
          for (h = 0; h < Nt_in; h++)
            for (j = 0; j < Ny_in; j++)
              for (i = 0; i < Nx_in; i++)
                {
                  Conc_in(h, 0, j, i) = Conc_in_tmp(h, 0, j, i);
                  for (k = 1; k < Nz_in; k++)
                    Conc_in(h, k, j, i) = Conc_in_tmp(h, k - 1, j, i);
                } // end i

          // conversion VMR (mole/mole) -> aerosol_density (ug/m3)
          // based on mozart4 model --> see: https://wiki.ucar.edu/display/mozart4/Using+MOZART-4+output
          // aerosol_density = VMR * P[Pa] / Rgas[J/K-kg] /T[K] * mw_aerosol[g/mole] /mw_air[g/mole] * 1E9[ug/kg]

          float mw_aer_ipol = weights_map.find(pos->second.species_in[l])->second;
          cout << ", convert (mw = " << mw_aer_ipol << ")";
          cout.flush();
          for (h = 0; h < Nt_in; h++)
            for (k = 0; k < Nz_in; k++)
              for (j = 0; j < Ny_in; j++)
                for (i = 0; i < Nx_in; i++)
                  Conc_in(h, k, j, i) *= Pressure(h, k, j, i) / Rgas / Temperature(h, k, j, i) * mw_aer_ipol / mw_air * 1000000000.;

          // Interpolations to output grid
          cout << ", interpolate" ;
          cout.flush();
          LinearInterpolationOneGeneral(Conc_in, Conc_out_x_tmp, 1);
          Conc_out_x_tmp.ThresholdMin(0.);
          LinearInterpolationOneGeneral(Conc_in, Conc_out_y_tmp, 1);
          Conc_out_y_tmp.ThresholdMin(0.);
          LinearInterpolationOneGeneral(Conc_in, Conc_out_z_tmp, 1);
          Conc_out_z_tmp.ThresholdMin(0.);

          for (h = 0; h < Nt_out; h++)
            {
              for (k = 0; k < Nz_out; k++)
                {
                  for (j = 0; j < Ny_out; j++)
                    for (i = 0; i < 2; i++)
                      Conc_out_x(h, k, j, i) += pos->second.ratio[l] * Conc_out_x_tmp(h, k, j, i);
                  for (j = 0; j < 2; j++)
                    for (i = 0; i < Nx_out; i++)
                      Conc_out_y(h, k, j, i) += pos->second.ratio[l] * Conc_out_y_tmp(h, k, j, i);
                } // end k

              for (k = 0; k < 1; k++)
                for (j = 0; j < Ny_out; j++)
                  for (i = 0; i < Nx_out; i++)
                    Conc_out_z(h, k, j, i) += pos->second.ratio[l] * Conc_out_z_tmp(h, k, j, i);

            } // end h
          cout << endl;
        } // end l

      // Set name for Polair3D species.
      string spec_size = pos->first ;

      string file_x = Directory_out + spec_size + "_x.bin";
      string file_y = Directory_out + spec_size + "_y.bin";
      string file_z = Directory_out + spec_size + "_z.bin";

      Polair.Append(Conc_out_x, file_x);
      Polair.Append(Conc_out_y, file_y);
      Polair.Append(Conc_out_z, file_z);

      cout << "in x direction (min: " << Conc_out_x.GetMin() << ")";
      cout << " (mean: " << Conc_out_x.Mean() << ")";
      cout << " (max: " << Conc_out_x.GetMax() << ")" << endl;
      cout << "in y direction (min: " << Conc_out_y.GetMin() << ")";
      cout << " (mean: " << Conc_out_y.Mean() << ")";
      cout << " (max: " << Conc_out_y.GetMax() << ")" << endl;
      cout << "in z direction (min: " << Conc_out_z.GetMin() << ")";
      cout << " (mean: " << Conc_out_z.Mean() << ")";
      cout << " (max: " << Conc_out_z.GetMax() << ")" << endl;

      cout << endl;
    } // end pos
  cout << "==============>>> DONE <<<==============" << endl;
  END;
  return 0;
} // end of main
