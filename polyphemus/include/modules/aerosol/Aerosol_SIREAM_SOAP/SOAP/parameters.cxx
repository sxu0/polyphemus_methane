
#ifndef PARAM_CXX
#define PARAM_CXX

#include "species.cxx"
#include "properties.hxx"

using namespace soap;
  
void system_coupling(model_config &config, vector<species>& surrogate)
{
  //Check if the system is coupled (at least a species condense on both phases) and give the
  // indexes of water and H+
  int n=surrogate.size();
  int i;
  config.iH2O=-1;
  config.coupled_phases=false;
  config.iNa=-1;
  config.iHSO4m=-1;
  config.iSO4mm=-1;
  config.iNO3m=-1;
  config.iClm=-1;
  config.iHp=-1;
  config.iNH4p=-1;

  for (i=0;i<n;++i)
    {
      if (surrogate[i].hydrophilic and surrogate[i].hydrophobic
          and surrogate[i].is_organic)
        config.coupled_phases=true;
      if (surrogate[i].name=="H2O")
        config.iH2O=i;
      if (surrogate[i].name=="H")
        config.iHp=i;
      if (surrogate[i].name=="HSO4")
        config.iHSO4m=i;
      if (surrogate[i].name=="SO4")
        config.iSO4mm=i;
      if (surrogate[i].name=="Cl")
        config.iClm=i;
      if (surrogate[i].name=="NH4")
        config.iNH4p=i;
      if (surrogate[i].name=="NO3")
        config.iNO3m=i;
      if (surrogate[i].name=="Na")
        config.iNa=i;

    }
}


void system_aiomfac(model_config &config, vector<species>& surrogate)
{
  //Create the matrixes for the aiomfac computation 
  //Parameters for the aiomfac computation of long ranges and medium ranges interactions
  // Value comes from Zuend et al. (ACP 2008) and Zuend et al. (ACP 2011)
  int i,j,k,l;
  int n=surrogate.size();
  int Ngroup_solute=15;
  int Nions=12; //H+ Li+ Na+ K+ NH4+ Mg2+ Ca2+ Cl- Br- NO3- HSO4- SO42-

  //Data:

  //     H+          Li+         Na+         K+         NH4+        Mg2+       Ca2+          Cl-       Br-          NO3-       HSO4-      SO42-
  double b1ki[15][12]={
    {9.44787e-2, 7.96162e-2, 1.05881e-1, 9.83642e-2, 5.66744e-2, 8.76190e-2, 1.02141e-1, 6.91431e-2, 4.13679e-2, 4.23323e-2, 1.24453e-1, 7.59843e-2}, //CHn
    {7.96996e-2, 6.18899e-2, 1.05881e-1, 6.84793e-2, 3.90465e-2, 6.74901e-2, 8.09257e-2, 5.22791e-2, 4.09489e-2, 3.00003e-2, 9.61748e-2, 5.25736e-2}, //CHn[OH]
    {-2.49640e-2,5.87773e-3, 7.65000e-3, 2.70478e-3, 5.23824e-3, 3.10304e-3, 1.03873e-2, 9.15639e-3, 4.68668e-3,-2.70727e-2,-6.10023e-2, 2.27122e-3}, //OH
    {-1.54168e-2,4.45875e-2, 5.55665e-2, 3.24047e-2,-2.53489e-2, 5.11236e-2, 2.60545e-2, 3.88547e-2, 2.99897e-2,-2.75568e-3, 7.99946e-2,-4.48204e-2}, //COOH
    {1.96849e-1, 1.11797e-1, 1.62860e-1, 1.33560e-1, 9.37140e-2, 1.44219e-1, 2.04458e-1, 1.32956e-1, 6.48040e-2, 4.54274e-2, 0.0       , 1.23888e-1}, //CHnCO
    {0.0       , 5.18983e-2, 1.05740e-1, 4.74863e-2, 5.62254e-2, 6.83111e-2, 1.02119e-1, 6.90488e-2, 3.71747e-2, 4.13984e-2, 0.0       , 3.90486e-2}, //CHO
    {9.35736e-2, 6.12871e-2, 1.01867e-1, 6.47475e-2, 4.06031e-2, 7.25245e-2, 8.25460e-2, 5.42575e-2, 4.13329e-2, 4.77384e-3, 6.07156e-2, 4.69045e-2}, //CHnO
    {0.0       , 1.11797e-1, 1.62860e-1, 1.01841e-1, 9.37140e-2, 0.0       , 2.01780e-1, 1.32798e-1, 6.47542e-2, 4.54273e-2, 0.0       , 1.72676e-2}, //CCOO
    {1.98174e-1, 1.28778e-1, 2.18720e-1, 1.59394e-1, 9.38185e-2, 1.46539e-1, 2.06213e-1, 1.32956e-1, 8.22912e-2, 6.70102e-2, 2.29945e-1, 1.24112e-1}, //C=C
    {9.06872e-2, 6.20802e-2, 1.05881e-1, 7.03087e-2, 5.58433e-2, 7.52276e-2, 8.10944e-2, 5.29042e-2, 4.12821e-2, 4.20941e-2, 9.63773e-2, 7.05645e-2}, //ACHn
    {5.50099e-2, 4.45902e-2, 7.50170e-2, 4.74863e-2, 4.06033e-2, 6.83111e-2, 7.10013e-2, 4.02855e-2, 2.99898e-2, 4.53537e-3, 0.0       , 3.66359e-2}, //ACOH
    {0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       }, //H2O
    {6.86096e-2, 6.71649e-2, 1.09517e-1, 6.74523e-2, 4.58414e-2, 7.56276e-2, 9.29332e-2, 6.34139e-2, 4.60196e-2,-2.22989e-2,-2.86662e-4, 4.91757e-2}, //CHnOOH
    {-1.54168e-2,4.45875e-2, 5.55665e-2, 3.24047e-2,-2.53489e-2, 5.11236e-2, 2.60545e-2, 3.88574e-2, 2.99897e-2,-2.75568e-3, 7.99946e-2,-4.48204e-2}, //C(=O)OOH
    {1.87147e-1, 1.22574e-1, 2.03734e-1, 1.29495e-1, 8.12062e-2, 1.45049e-1, 1.65092e-1, 1.08515e-1, 8.26657e-2, 9.54767e-3, 1.21431e-1, 9.38090e-2}};//CHnOOCHn

  double b2ki[15][12]={
    {6.56897e-2, 4.66958e-2, 2.26682e-2, 4.20328e-2, 4.31654e-2,-2.31755e-2, 7.47421e-2, 5.88900e-2, 3.43938e-2, 4.85325e-2, 1.42679e-1, 1.13096e-1}, //CHn
    {3.13421e-5, 2.58311e-2, 9.73093e-3, 3.20716e-2, 4.44483e-2,-2.23263e-2, 5.92614e-2, 3.65601e-2, 1.68833e-5, 3.96702e-2, 9.75711e-2, 7.93153e-2}, //CHn[OH]
    {7.78687e-5,-5.62998e-2,-7.48582e-3,-3.23740e-2, 5.68411e-3,-8.65932e-3,-5.46599e-4,-2.24924e-2, 1.47618e-3, 8.32695e-3,-3.95218e-2, 1.22439e-2}, //OH
    {-7.47215e-2,-6.96806e-3,-6.69604e-4,1.99525e-2,-3.14902e-2,-2.49540e-2, 2.33172e-2, 3.00968e-2, 1.23743e-2, 6.66352e-4, 5.62030e-6,-5.00144e-2}, //COOH
    {-4.27577e-3,1.42257e-1, 5.04545e-2, 5.03945e-2, 5.16922e-2,-2.09508e-2, 1.52489e-1, 1.36387e-1, 4.15174e-2, 6.62415e-2, 0.0       , 1.23888e-1}, //CHnCO
    { 0.0      ,-1.68243e-3, 2.27265e-2, 1.42102e-2, 3.70256e-2,-4.26791e-2, 7.47142e-2, 5.88674e-2, 1.80747e-2, 4.84721e-2, 0.0       , 7.19691e-2}, //CHO
    {6.14536e-2, 2.69698e-2, 1.48542e-2, 6.47239e-2, 4.50589e-2,-2.34403e-2, 7.07754e-2, 4.16149e-2, 2.39118e-2, 8.52485e-2, 2.43957e-2, 8.60605e-2}, //CHnO
    { 0.0      , 1.42257e-1, 5.04545e-2,-4.26811e-3, 5.16922e-2, 0.0       ,-1.07750e-1, 1.36298e-1, 4.15094e-2, 6.62415e-2, 0.0       , 1.07737e-3}, //CCOO
    {1.38151e-1, 1.36276e-1, 5.09909e-2, 6.73063e-2, 1.16555e-1,-1.39578e-2, 1.52563e-1, 1.36387e-1, 5.60745e-2, 1.23029e-1, 1.96250e-1, 2.71725e-1}, //C=C
    {2.23066e-2, 3.08726e-2, 2.26682e-2, 3.24520e-2, 4.29986e-2,-2.88362e-2, 5.93145e-2, 3.86039e-2, 3.40167e-2, 4.84843e-2, 9.77476e-2, 6.33234e-2}, //ACHn
    {1.41345e-3,-6.96568e-3, 6.85449e-3, 1.42102e-2, 2.12082e-2,-4.26791e-2, 2.88578e-2, 3.76714e-2, 1.23743e-2, 2.88017e-2, 0.0       , 6.14349e-2}, //ACOH
    {0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       }, //H2O
    {6.15315e-2,-2.93300e-2, 7.36838e-3, 3.23499e-2, 5.07430e-2,-3.20997e-2, 7.02288e-2, 1.91224e-2, 2.53879e-2, 9.35755e-2,-1.51261e-2, 9.38044e-2}, //CHnOOH
    {-7.47215e-5,-6.96806e-3,-6.69604e-4,1.99525e-2,-3.14902e-2,-2.49540e-2, 2.33172e-2, 3.00968e-2, 1.23743e-2, 6.66352e-4, 5.62030e-6,-5.00144e-2}};//CHnOOCHn

  double b1ca[12][12]={
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.182003  , 0.120325  , 0.210638  , 0.0215532 , 0.286343  }, //H+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 1.106555  , 0.106384  , 0.076313  , 0.0       , 0.114470  }, //Li+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.053741  , 0.180807  , 0.001164  , 0.0153214 , 0.001891  }, //Na+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.016561  , 0.033688  , 0.000025  , 0.0       , 0.004079  }, //K+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.001520  , 0.002498  ,-0.000057  , 0.00759735, 0.000373  }, //NH4+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.195909  , 0.0260487 , 0.430671  , 0.0       , 0.122364  }, //Mg2+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.104920  , 0.890929  , 0.163282  , 0.0       , 1.295670  }, //Ca2+
    { 0.182003 , 1.106555  , 0.053741  , 0.016561  , 0.001520  , 0.195909  , 0.104920  , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       }, //Cl-
    { 0.120325 , 0.106384  , 0.180807  , 0.033688  , 0.002498  , 0.0260487 , 0.890929  , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       }, //Br-
    { 0.210638 , 0.076313  , 0.001164  , 0.000025  ,-0.000057  , 0.430671  , 0.163282  , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       }, //NO3-
    { 0.0215532, 0.0       , 0.0153214 , 0.0       , 0.00759735, 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       }, //HSO4-
    { 0.286343 , 0.114470  , 0.001891  , 0.004079  , 0.000373  ,0.122364   , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       , 0.0       }};//SO4--

  
  double b2ca[12][12]={
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.243340  , 0.444859  , 0.122694  , 0.562966  , -5.99615  }, //H+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.206370  , 0.316480  , 0.300550  , 0.0       , 0.035401  }, //Li+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.079771  , 0.273114  , -0.102546 , 0.400000  , -0.424184 }, //Na+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , -0.002752 , 0.060882  , -0.413172 , 0.0       , -0.869936 }, //K+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.049074  , 0.081512  , -0.171746 , 0.143012  , -0.906075 }, //NH4+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.332387  , 1.017040  , 0.767242  , 0.0       , -3.425876 }, //Mg2+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.866923  , 0.0610134 , 0.203681  , 0.0       , -0.696806 }, //Ca2+
    { 0.243340 , 0.206370  , 0.079771  , -0.002752 , 0.049074  , 0.332387  , 0.866923  ,  0.0      ,  0.0      , 0.0       , 0.0       , 0.0       }, //Cl-
    { 0.444859 , 0.316480  , 0.273114  , 0.060882  , 0.081512  , 1.017040  , 0.0610134 ,  0.0      ,  0.0      , 0.0       , 0.0       , 0.0       }, //Br-
    { 0.122694 , 0.300550  , -0.102546 , -0.413172 , -0.171746 , 0.767242  , 0.203681  ,  0.0      ,  0.0      , 0.0       , 0.0       , 0.0       }, //NO3-
    { 0.562966 , 0.0       , 0.400000  , 0.0       , 0.143012  , 0.0       , 0.0       ,  0.0      ,  0.0      , 0.0       , 0.0       , 0.0       }, //HSO4-
    { -5.99615 , 0.035401  , -0.424184 , -0.869936 , -0.906075 , -3.425876 , -0.696806 ,  0.0      ,  0.0      , 0.0       , 0.0       , 0.0       }}; //SO4--
	  
  double b3ca[12][12];
  for (i=0; i<Nions; ++i)
    for (j=0; j<Nions; ++j)
      b3ca[i][j]=0.8;

  b3ca[0][10]=0.142442; //H+ HSO4-
  b3ca[10][0]=0.142442;
  b3ca[0][11]=1.36861;  //H+ SO4--
  b3ca[11][0]=1.36861;
  b3ca[2][9]=0.410453;  //Na+ NO3-
  b3ca[9][2]=0.410453;
  b3ca[2][10]=0.423635; //Na+ HSO4-
  b3ca[10][2]=0.423635;
  b3ca[3][9]=0.357227;  //K+  NO3-
  b3ca[9][3]=0.357227;
  b3ca[4][7]=0.116801;  //NH4+ Cl- 
  b3ca[7][4]=0.116801;
  b3ca[4][8]=0.143621;  //NH4+ Br-
  b3ca[8][4]=0.143621;
  b3ca[4][9]=0.260000;  //NH4+ NO3-
  b3ca[9][4]=0.260000;  
  b3ca[4][10]=0.203954; //NH4+ HSO4-
  b3ca[10][4]=0.203954; 
  b3ca[4][11]=0.545109; //NH4+ SO4--
  b3ca[11][4]=0.545109;

  double c1ca[12][12]={
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.033319  , 0.080767  , -0.101736 , 0.0703842  , -0.535977  }, //H+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.053239  , 0.057602  , 0.046701  , 0.0        , -0.263258  }, //Li+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.024553  , -0.506598 , 0.002535  , 0.00350072 , -0.223851  }, //Na+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.020833  , 0.015293  , -0.000455 , 0.0        , -0.092240  }, //K+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.011112  , 0.013795  , 0.005510  , 0.00631184 , -0.000379  }, //NH4+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.072063  , 0.061264  , -0.511836 , 0.0        , -0.738561  }, //Mg2+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.072063  , -0.238788 , -0.075452 , 0.0        , 1.59159    }, //Ca2+
    { 0.033319 , 0.053239  , 0.024553  , 0.020833  , 0.011112  , 0.072063  , 0.072063  ,  0.0      ,  0.0      , 0.0       , 0.0        , 0.0        }, //Cl-
    { 0.080767 , 0.057602  , -0.506598 , 0.015293  , 0.013795  , 0.061264  , -0.238788 ,  0.0      ,  0.0      , 0.0       , 0.0        , 0.0        }, //Br-
    { -0.101736, 0.046701  , 0.002535  , -0.000455 , 0.005510  , -0.511836 , -0.075452 ,  0.0      ,  0.0      , 0.0       , 0.0        , 0.0        }, //NO3-
    { 0.0703842,   0.0     , 0.00350072, 0.0       , 0.00631184, 0.0       , 0.0       ,  0.0      ,  0.0      , 0.0       , 0.0        , 0.0        }, //HSO4-
    { -0.535977, -0.263258 , -0.223851 , -0.092240 , -0.000379 , -0.738561 , 1.59159   ,  0.0      ,  0.0      , 0.0       , 0.0        , 0.0        }}; //SO4--

  double c2ca[12][12]={
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.504672  , 0.596776  , 1.676420  , 0.714194   , 0.907200   }, //H+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.535548  , 0.464658  , 0.664928  , 0.0        , 1.316967   }, //Li+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.562981  , 2.209050  , 0.512657  , 0.400000   , 1.053620   }, //Na+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.670530  , 0.565063  , 0.242244  , 0.0        , 0.918743   }, //K+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.653256  , 0.728984  , 0.529762  , 0.825386   , 0.354206   }, //NH4+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.397920  , 0.299475  , 1.440940  , 0.0        , 0.864380   }, //Mg2+
    { 0.0      ,   0.0     ,  0.0      , 0.0       , 0.0       , 0.0       , 0.0       , 0.365747  , 0.762961  , 1.210906  , 0.0        , 0.256217   }, //Ca2+
    {0.504672  , 0.535548  , 0.562981  , 0.670530  , 0.653256  , 0.397920  , 0.365747  ,  0.0      ,  0.0      , 0.0       , 0.0        , 0.0        }, //Cl-
    {0.596776  , 0.464658  , 2.209050  , 0.565063  , 0.728984  , 0.299475  , 0.762961  ,  0.0      ,  0.0      , 0.0       , 0.0        , 0.0        }, //Br-
    {1.676420  , 0.664928  , 0.512657  , 0.242244  , 0.529762  , 1.440940  , 1.210906  ,  0.0      ,  0.0      , 0.0       , 0.0        , 0.0        }, //NO3-
    {0.714194  ,   0.0     , 0.400000  , 0.0       , 0.825386  , 0.0       , 0.0       ,  0.0      ,  0.0      , 0.0       , 0.0        , 0.0        }, //HSO4-
    {0.907200  , 1.316967  , 1.053620  , 0.918743  , 0.354206  , 0.864380  , 0.256217  ,  0.0      ,  0.0      , 0.0       , 0.0        , 0.0        }}; //SO4--

  double Rcc[12][12],Qcca[12][12][12];
  for (i=0; i<Nions; ++i)
    for (j=0; j<Nions; ++j)
      Rcc[i][j]=0.0;

  Rcc[4][0]=-0.154486;
  Rcc[0][4]=Rcc[4][0];
  
  for (i=0; i<Nions; ++i)
    for (j=0; j<Nions; ++j)
      for (k=0; k<Nions; ++k)
        Qcca[i][j][k]=0.0;

  Qcca[4][0][10]=4.48354e-4;
  Qcca[0][4][10]=Qcca[4][0][10];

  //equivalence between aiomfac and unifac groups
  int Group_unifac_to_aiomfac[45]={0,0,0,0,
                                   1,1,1,1,
                                   1,1,1,1,
				   0,0,0,0,
                                   8,8,8,8,8,
                                   9,9,
                                   9,9,9,
                                   2,
                                   11,
                                   10,
                                   4,4,
                                   5,
                                   7,7,
                                   6,6,6,
                                   3,
                                   0,
                                   0,0,0,
                                   12,12,12};

  //molar masses of unifac groups
  double molar_mass_groups_unifac[45]={15.0,14.0,13.0,12.0,
                                       15.0,14.0,13.0,12.0,
                                       15.0,14.0,13.0,12.0,
				       15.0,14.0,13.0,12.0,
                                       27.0,26.0,26.0,25.0,24.0,
                                       13.0,12.0,
                                       27.0,26.0,25.0,
                                       17.0,
                                       18.0,
                                       29.0,
                                       43.0,42.0,
                                       29.0,
                                       59.0,58.0,
                                       31.0,30.0,29.0,
                                       45.0,
                                       58.0,
                                       76.0,75.0,74.0,
                                       47.0,46.0,45.0};

  //default structure (used if there is no group in the given structure of the molecule)
  double default_structure[45]={0.69,12.74,0.0,0.0, // group C
                                0.0,0.0,0.0,0.0, // group C[OH] 
                                0.0,0.0,0.0,0.0, // group Clacohol
				0.0,0.0,0.0,0.0, // group Clacohol-tail
                                0.0,0.05,0.0,0.0,0.0, //group C=C
                                1.42,0.86, //group aromatic carbon (AC)
                                0.0,0.15,0.0, // group //AC-C
                                0.0,  //group OH
                                0.0, //group H2O
                                0.15, //group ACOH
                                0.15,0.0, //group ketone
                                0.0,   //group aldehyde  
                                0.0,0.0, //group ester
                                0.30,0.0,0.0, //group ether
                                1.01,  //group acid
                                0.0,   //group ACNO2
                                0.0,0.0,0.0, //group NO3
                                0.0,0.0,0.0}; //group CO-OH


  //modification of the matrixes to keep only necessary informations:
  //      - remove unused solvent groups or ions groups
  double sum_group_solute[Ngroup_solute];
  double sum_group_species;
  int NFUNC=45;
  bool composition_already_defined;
  int index=0;
  int idefault=-1;
  
  
  for (j=0;j<Ngroup_solute;++j)
    sum_group_solute[j] = 0.0;
  
  for (i=0;i<n;++i)
    if (surrogate[i].hydrophilic and surrogate[i].compute_gamma_aq)
      {
        sum_group_species = 0.0;
        for (j=0;j<NFUNC;++j)
          if (Group_unifac_to_aiomfac[j]>=0)
            {
              sum_group_species += surrogate[i].groups[j];
              sum_group_solute[Group_unifac_to_aiomfac[j]] += surrogate[i].groups[j];
            }
		
        if (sum_group_species == 0.0)
          {
            if (idefault < 0)
              {
                idefault = i;
                surrogate[i].index_gamma_aiomfac = index;
                ++index;
                for (j=0;j<NFUNC;++j)
                  if (Group_unifac_to_aiomfac[j]>=0)
                    sum_group_solute[Group_unifac_to_aiomfac[j]] += default_structure[j];
              }
            else
              surrogate[i].index_gamma_aiomfac = surrogate[idefault].index_gamma_aiomfac;
          }
        else
          {
            if (i > 0 and index > 0)
              {
                composition_already_defined=false;
                for (k=0;k<i;++k)
                  if (surrogate[k].hydrophilic and surrogate[k].compute_gamma_aq
                      and composition_already_defined==false)
                    {
                      composition_already_defined=true;
                      for (j=0;j<NFUNC;++j)
                        if (Group_unifac_to_aiomfac[j]>=0)
                          if (surrogate[i].groups[j]
                              != surrogate[k].groups[j])
                            composition_already_defined=false;
                      if (composition_already_defined)
                        surrogate[i].index_gamma_aiomfac=surrogate[k].index_gamma_aiomfac;
                    }
                if (composition_already_defined==false)
                  {
                    surrogate[i].index_gamma_aiomfac = index;
                    ++index;
                  }
              }
            else
              {
                surrogate[i].index_gamma_aiomfac = index;
                ++index;
              }
          }
	  
      }
    else
      surrogate[i].index_gamma_aiomfac = -1;

  config.nmol_aiomfac=index;
  config.ngroup_aiomfac=0;
  for (j=0;j<Ngroup_solute;++j)
    if (sum_group_solute[j]>0.0)
      config.ngroup_aiomfac++;
  
  Array<int, 1> index_group_aiomfac;
  Array<int, 1> i_ion_aiomfac;
  index_group_aiomfac.resize(Ngroup_solute);
  i_ion_aiomfac.resize(Nions);

  index=0;
  
  for (j=0;j<Ngroup_solute;++j)
    if (sum_group_solute[j]>0.0)
      {
        index_group_aiomfac(j)=index;
        ++index;
      }
    else
      index_group_aiomfac(j)=-1;

  for (i=0;i<Nions;++i)
    i_ion_aiomfac(i)=-1;

  
  config.nion_aiomfac=0;
  for (i=0;i<n;++i)
    if (surrogate[i].is_organic==false and config.iH2O!=i and surrogate[i].is_inorganic_precursor==false)
      {	
        surrogate[i].index_ion=config.nion_aiomfac;
        i_ion_aiomfac(surrogate[i].index_ion_aiomfac)=surrogate[i].index_ion;
        config.nion_aiomfac++;
      }
    else
      surrogate[i].index_ion=-1;

  if (config.nmol_aiomfac>0 and config.nion_aiomfac>0)
    {
      config.b1ki_aq.resize(config.ngroup_aiomfac,config.nion_aiomfac);
      config.b2ki_aq.resize(config.ngroup_aiomfac,config.nion_aiomfac);
      config.b1ca_aq.resize(config.nion_aiomfac,config.nion_aiomfac);
      config.b2ca_aq.resize(config.nion_aiomfac,config.nion_aiomfac);
      config.b3ca_aq.resize(config.nion_aiomfac,config.nion_aiomfac);
      config.c1ca_aq.resize(config.nion_aiomfac,config.nion_aiomfac);
      config.c2ca_aq.resize(config.nion_aiomfac,config.nion_aiomfac);
      config.Rcc_aq.resize(config.nion_aiomfac,config.nion_aiomfac);
      config.Qcca_aq.resize(config.nion_aiomfac,config.nion_aiomfac,config.nion_aiomfac);
      config.groups_aiomfac.resize(config.nmol_aiomfac,config.ngroup_aiomfac);

      for (i=0;i<config.nmol_aiomfac;++i)
        for (j=0;j<config.ngroup_aiomfac;++j)
          config.groups_aiomfac(i,j)=0.0;

	  
      for (i=0;i<n;++i)
        if (surrogate[i].hydrophilic and surrogate[i].compute_gamma_aq and
            surrogate[i].index_gamma_aiomfac>=0)
          {
            sum_group_species = 0.0;	
			
            for (j=0;j<NFUNC;++j)
              if (Group_unifac_to_aiomfac[j]>=0)
                sum_group_species += surrogate[i].groups[j];
			
            double temp=0.0;
            for (j=0;j<config.ngroup_aiomfac;++j)
              temp+=config.groups_aiomfac(surrogate[i].index_gamma_aiomfac,j);

			
            if (temp==0.0)
              for (j=0;j<NFUNC;++j)
                if (Group_unifac_to_aiomfac[j]>=0)
                  if (sum_group_solute[Group_unifac_to_aiomfac[j]]>0.0)
                    {
                      if (sum_group_species==0.0)
                        config.groups_aiomfac(surrogate[i].index_gamma_aiomfac,
                                              index_group_aiomfac(Group_unifac_to_aiomfac[j]))+=
                          default_structure[j];
                      else
                        config.groups_aiomfac(surrogate[i].index_gamma_aiomfac,
                                              index_group_aiomfac(Group_unifac_to_aiomfac[j]))+=
                          surrogate[i].groups[j];
                    }
          }

      for (j=0;j<Ngroup_solute;++j)
        if (index_group_aiomfac(j)>=0)
          for (i=0;i<n;++i)
            if (surrogate[i].index_ion>=0)
              {
                config.b1ki_aq(index_group_aiomfac(j),surrogate[i].index_ion)=
                  b1ki[j][surrogate[i].index_ion_aiomfac];
                config.b2ki_aq(index_group_aiomfac(j),surrogate[i].index_ion)=
                  b2ki[j][surrogate[i].index_ion_aiomfac];
				
              }
	  
      for (j=0;j<Nions;++j)
        if (i_ion_aiomfac(j)>=0)
          for (k=0;k<Nions;++k)
            if (i_ion_aiomfac(k)>=0)
              {
                config.b1ca_aq(i_ion_aiomfac(j),i_ion_aiomfac(k))=b1ca[j][k];
                config.b2ca_aq(i_ion_aiomfac(j),i_ion_aiomfac(k))=b2ca[j][k];
                config.b3ca_aq(i_ion_aiomfac(j),i_ion_aiomfac(k))=b3ca[j][k];
                config.c1ca_aq(i_ion_aiomfac(j),i_ion_aiomfac(k))=c1ca[j][k];
                config.c2ca_aq(i_ion_aiomfac(j),i_ion_aiomfac(k))=c2ca[j][k];
                config.Rcc_aq(i_ion_aiomfac(j),i_ion_aiomfac(k))=Rcc[j][k];
                for (l=0;l<Nions;++l)
                  if (i_ion_aiomfac(l)>=0)
                    config.Qcca_aq(i_ion_aiomfac(j),i_ion_aiomfac(k),
                                   i_ion_aiomfac(l))=Qcca[j][k][l];
              }
    }

  config.Mgroups_aiomfac.resize(config.nmol_aiomfac,config.ngroup_aiomfac);
  for (i=0;i<config.nmol_aiomfac;++i)
    for (j=0;j<config.ngroup_aiomfac;++j)
      config.Mgroups_aiomfac(i,j)=0.0;
  
  for (i=0;i<n;++i)
    if (surrogate[i].hydrophilic and surrogate[i].index_gamma_aiomfac>=0.0)
      if (i==idefault)
        {
	  for (j=0;j<NFUNC;++j)
	    if (index_group_aiomfac(Group_unifac_to_aiomfac[j])>=0 and
                config.groups_aiomfac(surrogate[i].index_gamma_aiomfac,
                                      index_group_aiomfac(Group_unifac_to_aiomfac[j]))>0.0)
	      config.Mgroups_aiomfac(surrogate[i].index_gamma_aiomfac,index_group_aiomfac(Group_unifac_to_aiomfac[j]))=0.0;

          for (j=0;j<NFUNC;++j)
            if (index_group_aiomfac(Group_unifac_to_aiomfac[j])>=0 and
                config.groups_aiomfac(surrogate[i].index_gamma_aiomfac,
                                      index_group_aiomfac(Group_unifac_to_aiomfac[j]))>0.0)
              config.Mgroups_aiomfac(surrogate[i].index_gamma_aiomfac,index_group_aiomfac(Group_unifac_to_aiomfac[j]))+=
                molar_mass_groups_unifac[j]*default_structure[j]/
                config.groups_aiomfac(surrogate[i].index_gamma_aiomfac,
                                      index_group_aiomfac(Group_unifac_to_aiomfac[j]));
        }
      else
	{
	  for (j=0;j<NFUNC;++j)
	    if (index_group_aiomfac(Group_unifac_to_aiomfac[j])>=0 and
		config.groups_aiomfac(surrogate[i].index_gamma_aiomfac,
				      index_group_aiomfac(Group_unifac_to_aiomfac[j]))>0.0)
	      config.Mgroups_aiomfac(surrogate[i].index_gamma_aiomfac,index_group_aiomfac(Group_unifac_to_aiomfac[j]))=0.0;

	  for (j=0;j<NFUNC;++j)
	    if (index_group_aiomfac(Group_unifac_to_aiomfac[j])>=0 and
		config.groups_aiomfac(surrogate[i].index_gamma_aiomfac,
				      index_group_aiomfac(Group_unifac_to_aiomfac[j]))>0.0)
	      {
		config.Mgroups_aiomfac(surrogate[i].index_gamma_aiomfac,index_group_aiomfac(Group_unifac_to_aiomfac[j]))+=
		  molar_mass_groups_unifac[j]*surrogate[i].groups[j]/
		  config.groups_aiomfac(surrogate[i].index_gamma_aiomfac,
					index_group_aiomfac(Group_unifac_to_aiomfac[j]));
	      }
	}

		
}

void param_unifac(model_config &config, vector<species> &surrogate)
{
  //Create the matrixes for the unifac computation 
  //Parameters for the unifac computation of short ranges interactions
  // Parameters comes from :
  //       http://www.aim.env.uea.ac.uk/aim/info/UNIFACgroups.html
  //       Compernolle et al. (ACP 2009)
  //       Marcolli et al. (ACP 2005)
  
  int NFUNC = 45;
  config.Z = 10.0;
  //                 CH3     CH2     CH      C      CH3[OH] CH2[0H]  CH[OH]  C[OH] CH3al   CH2alc   CHalc   Calc    CH3alct   CH2alct   CHalct   Calct   CH2=CH   CH=CH   CH2=C  CH=C     C=C      ACH    AC      ACCH3   ACCH2   ACCH    OH     H2O   ACOH    CH3CO   CH2CO   HCO    CH3COO   CH2COO  CH3O    CH2O    CHO     COOH   ACNO2  CH2ONO2 CHONO2  CONO2    CH2O-OH  CHO-OH   CO-OH
  double RG[45] = {0.9011, 0.6744, 0.4469, 0.2195, 0.9011, 0.6744, 0.4469, 0.2195, 0.9011, 0.6744, 0.4469, 0.2195, 0.9011, 0.6744, 0.4469, 0.2195, 1.3454, 1.1167, 1.1173, 0.8886, 0.6605, 0.5313, 0.3652, 1.2663, 1.0396, 0.8121, 1.0000, 0.92, 0.8952, 1.6724, 1.4457, 0.9980, 1.9031, 1.6764, 1.1450, 0.9183, 0.6908, 1.3013, 1.4199, 2.1246, 1.8971, 1.6697, 1.5869, 1.3594, 1.1320};
  double QG[45] = {0.8480, 0.5400, 0.2280, 0.0000, 0.8480, 0.5400, 0.2280, 0.0000, 0.8480, 0.5400, 0.2280, 0.0000, 0.8480, 0.5400, 0.2280, 0.0000, 1.1760, 0.8670, 0.9880, 0.6760, 0.4850, 0.4000, 0.1200, 0.9680, 0.6600, 0.3480, 1.2000, 1.40, 0.6800, 1.4480, 1.1800, 0.9480, 1.7280, 1.4200, 1.0880, 0.7800, 0.4680, 1.2240, 1.1040, 1.8682, 1.5562, 1.3282, 1.4370, 1.1250, 0.8970};

  //                    H+    Li+   Na+   K+   NH4+  Mg2+  Ca2+  Cl-   Br-   NO3-  HSO4-  SO4--
  double RGions[12] = {1.78, 0.61, 0.38, 0.44, 0.69, 5.44, 2.24, 0.99, 1.25, 0.95, 1.65,  3.34};
  double QGions[12] = {2.70, 0.99, 0.62, 0.58, 0.78, 8.35, 3.40, 0.99, 1.16, 0.97, 1.40,  3.96};
    

  //matrix of interactions between groups
  // CH3     CH2     CH      C      CH3[OH] CH2[0H]  CH[OH]  C[OH] CH3alc   CH2alc   CHalc   Calc    CH3alct   CH2alct   CHalct   Calct   CH2=CH   CH=CH   CH2=C  CH=C     C=C      ACH    AC    ACCH3   ACCH2  ACCH   OH     H2O   ACOH    CH3CO   CH2CO   HCO    CH3COO   CH2COO  CH3O    CH2O    CHO     COOH   ACNO2  CH2ONO2 CHONO2  CONO2    CH2O-OH  CHO-OH   CO-OH
  double A[45][45] = {
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 1318.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //CH3
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 1318.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //CH2
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 1318.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //CH
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 1318.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //C
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 2314.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //CH3[OH]
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 2314.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //CH2[OH]
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 2314.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //CH[OH]
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 2314.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //C[OH]
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 1890.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //CH3alc
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 1890.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //CH2alc
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 1890.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //CHalc
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 1890.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //Calc
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 1325.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //CH3alct
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 1325.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //CH2alct
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 1325.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //CHalct
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 86.020, 86.020, 86.020, 86.020, 86.020, 61.13, 61.13, 76.50, 76.50, 76.50, 986.50, 1325.0, 1333.0, 476.40, 476.40, 677.0, 232.10, 232.10, 251.50, 251.50, 251.50, 663.50,  543.0, 500.95, 500.95, 500.95,  977.56, 977.56, 977.56}, //Calct
    {-35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36,   -35.36,    -35.36, -35.36, -35.36 ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 38.81, 38.81, 74.15, 74.15, 74.15, 524.10, 270.60, 526.10, 182.60, 182.60, 448.8, 37.850, 37.850, 214.50, 214.50, 214.50, 318.90,  0.0  ,10326.0,10326.0,10326.0,  475.91, 475.91, 475.91}, //CH2=CH interactions with inorganic ions missing
    {-35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36,   -35.36,    -35.36, -35.36, -35.36 ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 38.81, 38.81, 74.15, 74.15, 74.15, 524.10, 270.60, 526.10, 182.60, 182.60, 448.8, 37.850, 37.850, 214.50, 214.50, 214.50, 318.90,  0.0  ,10326.0,10326.0,10326.0,  475.91, 475.91, 475.91}, //CH2=C interactions with inorganic ions missing
    {-35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36,   -35.36,    -35.36, -35.36, -35.36 ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 38.81, 38.81, 74.15, 74.15, 74.15, 524.10, 270.60, 526.10, 182.60, 182.60, 448.8, 37.850, 37.850, 214.50, 214.50, 214.50, 318.90,  0.0  ,10326.0,10326.0,10326.0,  475.91, 475.91, 475.91}, //CH2=C interactions with inorganic ions missing
    {-35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36,   -35.36,    -35.36, -35.36, -35.36 ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 38.81, 38.81, 74.15, 74.15, 74.15, 524.10, 270.60, 526.10, 182.60, 182.60, 448.8, 37.850, 37.850, 214.50, 214.50, 214.50, 318.90,  0.0  ,10326.0,10326.0,10326.0,  475.91, 475.91, 475.91}, //CH=C interactions with inorganic ions missing
    {-35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36, -35.36,   -35.36,    -35.36, -35.36, -35.36 ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 38.81, 38.81, 74.15, 74.15, 74.15, 524.10, 270.60, 526.10, 182.60, 182.60, 448.8, 37.850, 37.850, 214.50, 214.50, 214.50, 318.90,  0.0  ,10326.0,10326.0,10326.0,  475.91, 475.91, 475.91}, //C=C interactions with inorganic ions missing
    {-11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, 3.4460, 3.4460, 3.4460, 3.4460, 3.4460, 0.0  , 0.0  , 167.0, 167.0, 167.0, 636.10, 903.80, 1329.0, 25.770, 25.770, 347.3, 5.9940, 5.9940, 32.140, 32.140, 32.140, 537.40,  194.9, 0.0   , 0.0   , 0.0   ,  0.0   ,  0.0  , 0.0   }, //ACH interactions with inorganic ions missing and nitrate and hydroperoxide groups are missing
    {-11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, -11.12, 3.4460, 3.4460, 3.4460, 3.4460, 3.4460, 0.0  , 0.0  , 167.0, 167.0, 167.0, 636.10, 903.80, 1329.0, 25.770, 25.770, 347.3, 5.9940, 5.9940, 32.140, 32.140, 32.140, 537.40,  194.9, 0.0   , 0.0   , 0.0   ,  0.0   ,  0.0  , 0.0   }, //AC interactions with inorganic ions missing and nitrate and hydroperoxide groups are missing
    {-69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -113.6, -113.6, -113.6, -113.6, -113.6,-146.8,-146.8, 0.0  , 0.0  , 0.0  , 803.20, 5695.0, 884.90, -52.10, -52.10, 586.6, 5688.0, 5688.0, 213.10, 213.10, 213.10, 872.30, 4448.0, 0.0   , 0.0   , 0.0   ,  0.0   ,  0.0  , 0.0   }, //ACCH3 interactions with inorganic ions missing and nitrate and hydroperoxide groups are missing
    {-69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -113.6, -113.6, -113.6, -113.6, -113.6,-146.8,-146.8, 0.0  , 0.0  , 0.0  , 803.20, 5695.0, 884.90, -52.10, -52.10, 586.6, 5688.0, 5688.0, 213.10, 213.10, 213.10, 872.30, 4448.0, 0.0   , 0.0   , 0.0   ,  0.0   ,  0.0  , 0.0   }, //ACCH2 interactions with inorganic ions missing and nitrate and hydroperoxide groups are missing
    {-69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -69.70, -113.6, -113.6, -113.6, -113.6, -113.6,-146.8,-146.8, 0.0  , 0.0  , 0.0  , 803.20, 5695.0, 884.90, -52.10, -52.10, 586.6, 5688.0, 5688.0, 213.10, 213.10, 213.10, 872.30, 4448.0, 0.0   , 0.0   , 0.0   ,  0.0   ,  0.0  , 0.0   }, //ACCH interactions with inorganic ions missing and nitrate and hydroperoxide groups are missing
    {156.40, 156.40, 156.40, 156.40, 156.40, 156.40, 156.40, 156.40, 156.40, 156.40, 156.40, 156.40, 156.40, 156.40, 156.40, 156.40, 457.00, 457.00, 457.00, 457.00, 457.00, 89.60, 89.60, 25.82, 25.82, 25.82, 0.0   , 276.40, -259.7, 84.00 , 84.00 ,-203.6, 101.10, 101.10, 28.060, 28.060, 28.060, 224.39, 157.10, 37.631, 37.631, 37.631, -330.28,-330.28,-330.28}, //OH
    {300.00, 300.00, 300.00, 300.00, -89.71, -89.71, -89.71, -89.71, 162.30, 162.30, 162.30, 162.30, 362.10, 362.10, 362.10, 362.10, 496.10, 496.10, 496.10, 496.10, 496.10, 362.3, 362.3, 377.6, 377.6, 377.6, -153.3, 0.0   , 324.50, -195.4, -195.4,-116.0, 72.870, 72.870, 540.50, 540.50, 540.50, -69.29, 399.50, 142.65, 142.65, 142.65, -341.18,-341.18,-341.18}, //H2O interactions with inorganic ions not taken into account
    {275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 217.50, 217.50, 217.50, 217.50, 217.50, 25.34, 25.34, 244.2, 244.2, 244.2, -451.6, -601.8, 0.0   ,-356.10,-356.10,-271.1, -449.4, -449.4, -162.9, -162.9, -162.9, 408.90,-413.48, 0.0   , 0.0   , 0.0   , 0.0    , 0.0   ,  0.0  }, //ACOH interactions with inorganic ions missing and nitrate and hydroperoxide groups are missing
    {26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 42.920, 42.920, 42.920, 42.920, 42.920, 140.1, 140.1, 365.8, 365.8, 365.8, 164.50, 472.50, -133.1, 0.0   , 0.0   ,-37.36, -213.7, -213.7, -103.6, -103.6, -103.6, 669.40, 548.50,-197.93,-197.93,-197.93, -350.58,-350.58,-350.58}, //CH3CO
    {26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 26.760, 42.920, 42.920, 42.920, 42.920, 42.920, 140.1, 140.1, 365.8, 365.8, 365.8, 164.50, 472.50, -133.1, 0.0   , 0.0   ,-37.36, -213.7, -213.7, -103.6, -103.6, -103.6, 669.40, 548.50,-197.93,-197.93,-197.93, -350.58,-350.58,-350.58}, //CH2CO
    {505.70, 505.70, 505.70, 505.70, 505.70, 505.70, 505.70, 505.70, 505.70, 505.70, 505.70, 505.70, 505.70, 505.70, 505.70, 505.70, 56.300, 56.300, 56.300, 56.300, 56.300, 23.39, 23.39, 106.0, 106.0, 106.0, 529.00, 480.80, -155.6, 128.00, 128.00, 0.0  , -110.3, -110.3, 304.10, 304.10, 304.10, 497.50, 0.0   , 402.00, 402.00, 402.00, -387.63,-387.63,-387.93}, //HCO interactions with inorganic ions set equal to the parameters of CH2CO groups	  
    {114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 132.10, 132.10, 132.10, 132.10, 132.10, 85.84, 85.84,-170.0,-170.0,-170.0, 245.40, 200.80, -36.72, 372.20, 372.20,185.10, 0.0   , 0.0   , -235.7, -235.7, -235.7, 660.20, 0.0   , 1273.8, 1273.8, 1273.8, 928.33 , 928.33, 928.33}, //CH3COO interactions with inorganic ions missing
    {114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 114.80, 132.10, 132.10, 132.10, 132.10, 132.10, 85.84, 85.84,-170.0,-170.0,-170.0, 245.40, 200.80, -36.72, 372.20, 372.20,185.10, 0.0   , 0.0   , -235.7, -235.7, -235.7, 660.20, 0.0   , 1273.8, 1273.8, 1273.8, 928.33 , 928.33, 928.33}, //CH2COO interactions with inorganic ions missin
    {83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 26.510, 26.510, 26.510, 26.510, 26.510, 52.13, 52.13, 65.69, 65.69, 65.69, 237.70, -314.7, -178.5, 191.10, 191.10,-7.838, 461.30, 461.30, 0.0   , 0.0   , 0.0   , 664.60, 155.11, 1133.1, 1133.1, 1133.1, -438.74,-438.74,-438.74}, //CH3O interactions with inorganic ions missing
    {83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 26.510, 26.510, 26.510, 26.510, 26.510, 52.13, 52.13, 65.69, 65.69, 65.69, 237.70, -314.7, -178.5, 191.10, 191.10,-7.838, 461.30, 461.30, 0.0   , 0.0   , 0.0   , 664.60, 155.11, 1133.1, 1133.1, 1133.1, -438.74,-438.74,-438.74}, //CH2O interactions with inorganic ions missing
    {83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 83.360, 26.510, 26.510, 26.510, 26.510, 26.510, 52.13, 52.13, 65.69, 65.69, 65.69, 237.70, -314.7, -178.5, 191.10, 191.10,-7.838, 461.30, 461.30, 0.0   , 0.0   , 0.0   , 664.60, 155.11, 1133.1, 1133.1, 1133.1, -438.74,-438.74,-438.74}, //CHO interactions with inorganic ions missing
    {315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 1264.0, 1264.0, 1264.0, 1264.0, 1264.0, 62.32, 62.32, 89.86, 89.86, 89.86,-103.03,-145.88, -11.00, -297.8, -297.8,-165.5, -256.3, -256.3, -338.5, -338.5, -338.5, 0.0   , 0.0   ,-100.17,-100.17,-100.17, -501.23,-501.23,-501.23}, //COOH
    {5541.0, 5541.0, 5541.0, 5541.0, 5541.0, 5541.0, 5541.0, 5541.0, 5541.0, 5541.0, 5541.0, 5541.0, 5541.0, 5541.0, 5541.0, 5541.0, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,1824.0,1824.0,-127.8,-127.8,-127.8, 561.60, 360.70, 815.12, -101.5, -101.5, 0.0  , 0.0   , 0.0   , 220.66, 220.66, 220.66, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   ,  0.0  }, //ACNO2 interactions with numerous groups are missing
    {-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-294.43,-294.43,-294.43,-294.43,-294.43, 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 818.97, 681.78, 0.0   , 188.72, 188.72,-179.38,-356.25,-356.25,-289.81,-289.81,-289.81,1173.3, 0.0   , 0.0   , 0.0   , 0.0   , 545.66 , 545.66, 545.66}, //CH2ONO2 interactions with several groups and inorganic ions are missing	
    {-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-294.43,-294.43,-294.43,-294.43,-294.43, 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 818.97, 681.78, 0.0   , 188.72, 188.72,-179.38,-356.25,-356.25,-289.81,-289.81,-289.81,1173.3, 0.0   , 0.0   , 0.0   , 0.0   , 545.66 , 545.66, 545.66}, //CHONO2 interactions with several groups and inorganic ions are missing
    {-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-75.718,-294.43,-294.43,-294.43,-294.43,-294.43, 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 818.97, 681.78, 0.0   , 188.72, 188.72,-179.38,-356.25,-356.25,-289.81,-289.81,-289.81,1173.3, 0.0   , 0.0   , 0.0   , 0.0   , 545.66 , 545.66, 545.66}, //CONO2 interactions with several groups and inorganic ions are missing
    {-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-57.949,-57.949,-57.949,-57.949,-57.949, 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 342.92, 795.55, 0.0   , 380.94, 380.94, 408.88, -355.0, -355.0, 490.36, 490.36, 490.36,1479.0, 0.0   ,-86.279,-86.279,-86.279, 0.0    , 0.0   , 0.0   }, //CH2O-OH interactions with several groups and inorganic ions are missing
    {-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-57.949,-57.949,-57.949,-57.949,-57.949, 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 342.92, 795.55, 0.0   , 380.94, 380.94, 408.88, -355.0, -355.0, 490.36, 490.36, 490.36,1479.0, 0.0   ,-86.279,-86.279,-86.279, 0.0    , 0.0   , 0.0   }, //CHO-OH interactions with several groups and inorganic ions are missing
    {-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-23.233,-57.949,-57.949,-57.949,-57.949,-57.949, 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 342.92, 795.55, 0.0   , 380.94, 380.94, 408.88, -355.0, -355.0, 490.36, 490.36, 490.36,1479.0, 0.0   ,-86.279,-86.279,-86.279, 0.0    , 0.0   , 0.0   }}; //CO-OH interactions with several groups and inorganic ions are missing

  double B[45][45] = {
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50, 1248.1, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH3
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50, 1248.1, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH2
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50, 1248.1, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50, 1248.1, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //C
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50,-2151.0, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH3[OH]
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50,-2151.0, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH2[OH]
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50,-2151.0, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH[OH]
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50,-2151.0, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //C[OH]
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50, 1890.0, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH3alc
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50, 1890.0, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH2alc
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50, 1890.0, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CHalc
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50, 1890.0, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //Calc
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50, 1325.0, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH3alct
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50, 1325.0, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH2alct
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50, 1325.0, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CHalct
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 986.50, 1325.0, 1333.0,-476.40,-476.40, 200.0,-20.762,-20.762,-179.43,-179.43,-179.43, 663.50,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //Calct
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH2=CH interactions with inorganic ions missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH2=C interactions with inorganic ions missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH=CH interactions with inorganic ions missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH=C interactions with inorganic ions missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //C=C interactions with inorganic ions missing
    {19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  ,-636.10, 903.80,-1329.0,-187.18,-187.18,-347.3, 0.0  , 0.0   , -12.963,-12.963,-12.963,-537.40,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   ,  0.0  , 0.0   }, //ACH interactions with inorganic ions missing and nitrate and hydroperoxide groups are missing
    {19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 19.504, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  ,-636.10, 903.80,-1329.0,-187.18,-187.18,-347.3, 0.0  , 0.0   , -12.963,-12.963,-12.963,-537.40,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   ,  0.0  , 0.0   }, //AC interactions with inorganic ions missing and nitrate and hydroperoxide groups are missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //ACCH3 interactions with inorganic ions missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //ACCH3 interactions with inorganic ions missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //ACCH3 interactions with inorganic ions missing
    {200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 200.0, 200.0, 0.0  , 0.0  , 0.0  , 0.0   , 276.40,  259.7, 198.47, 198.47,-203.6, 200.00, 200.00,-128.99,-128.99,-128.99, 224.39, 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0     , 0.0   }, //OH
    {2.07691,2.07691,2.07691,2.07691,-200.00,-200.00,-200.00,-200.00,-200.00,-200.00,-200.00,-200.00,-173.73,-173.73,-173.73,-173.73, 0.0  , 0.0   , 0.0   , 0.0   , 0.0   ,-362.3,-362.3, 0.0  , 0.0  , 0.0  , 100.86, 0.0   , 264.56, 45.845, 45.845,-29.366,-13.512,-13.512, 355.03, 355.03, 355.03,-140.82, 0.0  , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   ,  0.0  }, //H2O interactions with inorganic ions not taken into account
    {275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 275.80, 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 200.00, 200.0, 0.0  , 0.0  , 0.0  ,  451.6, 98.265, 0.0   , 356.10, 356.10, 0.0  , 356.89, 356.89, 0.0   , 0.0   , 0.0  , 408.90,   0.0  , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   ,  0.0  },  //ACOH interactions with inorganic ions missing and nitrate and hydroperoxide groups are missing
    {-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597, 0.0  , 0.0   , 0.0  , 0.0   , 0.0   ,-189.62,-189.62,0.0  , 0.0  , 0.0  , 200.00, 114.60, 200.00, 0.0   , 0.0   , 0.0  , 0.0   , 213.70, 0.0   , 0.0   , 0.0  , -669.35,  0.0  , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   ,  0.0  }, //CH3CO
    {-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597,-50.597, 0.0  , 0.0   , 0.0  , 0.0   , 0.0   ,-189.62,-189.62,0.0  , 0.0  , 0.0  , 200.00, 114.60, 200.00, 0.0   , 0.0   , 0.0  , 0.0   , 213.70, 0.0   , 0.0   , 0.0  , -669.35,  0.0  , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   ,  0.0  }, //CH2CO
    {505.69, 505.69, 505.69, 505.69, 505.69, 505.69, 505.69, 505.69, 505.69, 505.69, 505.69, 505.69, 505.69, 505.69, 505.69, 505.69, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,-200.0,-200.0, 0.0  , 0.0  , 0.0  ,-529.00, 480.80, 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  ,-497.50, 0.0   ,  0.0   , 0.0   , 0.0   , 0.0    , 0.0   ,  0.0  }, //HCO interactions with inorganic ions set equal to the parameters of CH2CO groups	  
    {200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 245.40, 77.173, 199.50,-116.27,-116.27, 0.0  , 0.0   , 0.0   , -217.7, -217.7, -217.7,-429.17, 0.0   , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   , 0.0  }, //CH3COO interactions with inorganic ions missing
    {200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 245.40, 77.173, 199.50,-116.27,-116.27, 0.0  , 0.0   , 0.0   , -217.7, -217.7, -217.7,-429.17, 0.0   , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   , 0.0  }, //CH2COO interactions with inorganic ions missing
    {200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,-200.00,-200.00, 0.0, 0.0  , 0.0  , 237.70, -314.7, 0.0   , 0.0   , 0.0   , 0.0  ,-459.06,-459.06, 0.0   , 0.0   , 0.0   ,-664.60, 0.0   , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   , 0.0  }, //CH3O interactions with inorganic ions missing
    {200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,-200.00,-200.00, 0.0, 0.0  , 0.0  , 237.70, -314.7, 0.0   , 0.0   , 0.0   , 0.0  ,-459.06,-459.06, 0.0   , 0.0   , 0.0   ,-664.60, 0.0   , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   , 0.0  }, //CH2O interactions with inorganic ions missing
    {200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 200.00, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,-200.00,-200.00, 0.0, 0.0  , 0.0  , 237.70, -314.7, 0.0   , 0.0   , 0.0   , 0.0  ,-459.06,-459.06, 0.0   , 0.0   , 0.0   ,-664.60, 0.0   , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   , 0.0  }, //CHO interactions with inorganic ions missing
    {315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 315.30, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,-4.4703,-4.4703, 0.0, 0.0  , 0.0  , 200.00, 47.584,200.00, -74.212,-74.214,20.332,  256.3,  256.3, -145.74, -145.74, -145.74, 0.0, 0.0   ,0.0   , 0.0   , 0.0   , 0.0    , 0.0   , 0.0  }, //COOH
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   },  //ACNO2 interactions with numerous groups are missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CHONO2 interactions with several groups and inorganic ions are missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CONO2 interactions with several groups and inorganic ions are missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH2O-OH interactions with several groups and inorganic ions are missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CHO-OH interactions with several groups and inorganic ions are missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }}; //CO-OH interactions with several groups and inorganic ions are missing

  double C[45][45] = {
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976, 5.2720, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH3
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976, 5.2720, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH2
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976, 5.2720, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976, 5.2720, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //C
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976, -1.6464, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH3[OH]
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976, -1.6464, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH2[OH]
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976, -1.6464, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH[OH]
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976, -1.6464, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //C[OH]
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976,  7.5600, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH3alc
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976,  7.5600, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH2alc
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976,  7.5600, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CHalc
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976,  7.5600, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //Calc
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976,  2.9603, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH3alct
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976,  2.9603, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH2alct
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976,  2.9603, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CHalct
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0    ,   0.0     ,   0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.30257,0.30257, 0.0  , 0.0  , 0.0  , 2.1976,  2.9603, 3.1475,1.9056,1.9056,0.42165,0.92840,0.92840,0.55434,0.55434,0.55434, 2.6540,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //Calct
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH2=CH interactions with inorganic ions missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH2=C interactions with inorganic ions missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH=CH interactions with inorganic ions missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH=C interactions with inorganic ions missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //C=C interactions with inorganic ions missing
    {0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  ,2.5444, 2.3228,-5.3160, 0.0,0.0, 0.0, 0.0  , 0.0   , 0.8,0.8,0.8,-0.14576,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   ,  0.0  , 0.0   }, //ACH interactions with inorganic ions missing and nitrate and hydroperoxide groups are missing
    {0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  ,2.5444, 2.3228,-5.3160, 0.0,0.0, 0.0, 0.0  , 0.0   , 0.8,0.8,0.8,-0.14576,  0.0  , 0.0   , 0.0   , 0.0   ,  0.0   ,  0.0  , 0.0   }, //AC interactions with inorganic ions missing and nitrate and hydroperoxide groups are missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //ACCH3 interactions with inorganic ions missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //ACCH2 interactions with inorganic ions missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //ACCH interactions with inorganic ions missing
    {0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.8, 0.8, 0.0  , 0.0  , 0.0  , 0.0   , -1.1056, 0.97596, 0.8, 0.8,-0.12530, 0.8, 0.8,-0.8,-0.8,-0.8, 0.89756, 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0     , 0.0   }, //OH
    {-1.2,-1.2,-1.2,-1.2,-0.8,-0.8,-0.8,-0.8,-0.79977,-0.79977,-0.79977,-0.79977,-1.4484,-1.4484,-1.4484,-1.4484, 0.0  , 0.0   , 0.0   , 0.0   , 0.0   ,1.2682,1.2682, 0.0  , 0.0  , 0.0  , -0.8, 0.0   , -1.2980, -0.8,-0.8,0.0,-0.8,-0.8, -1.8480,-1.8480,-1.8480,-0.8, 0.0  , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   ,  0.0  }, //H2O interactions with inorganic ions not taken into account
    {0.97679, 0.97679, 0.97679, 0.97679, 0.97679, 0.97679, 0.97679, 0.97679, 0.97679, 0.97679, 0.97679, 0.97679, 0.97679, 0.97679, 0.97679, 0.97679, 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , -0.49742,-0.49742, 0.0  , 0.0  , 0.0  ,0.18064,-0.20012, 0.0   , 0.0, 0.0, 0.0  , 0.0, 0.0, 0.0   , 0.0   , 0.0  , 0.17894,   0.0  , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   ,  0.0  },  //ACOH interactions with inorganic ions missing and nitrate and hydroperoxide groups are missing
    {-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8, 0.0  , 0.0   , 0.0  , 0.0   , 0.0   ,0.0,0.0,0.0  , 0.0  , 0.0  , 0.8, -1.89, 0.0, 0.0   , 0.0   , 0.0  , 0.0   , 0.0, 0.0   , 0.0   , 0.0  , -2.6776,  0.0  , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   ,  0.0  }, //CH3CO
    {-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8, 0.0  , 0.0   , 0.0  , 0.0   , 0.0   ,0.0,0.0,0.0  , 0.0  , 0.0  , 0.8, -1.89, 0.0, 0.0   , 0.0   , 0.0  , 0.0   , 0.0, 0.0   , 0.0   , 0.0  , -2.6776,  0.0  , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   ,  0.0  }, //CH2CO
    {-2.0228, -2.0228, -2.0228, -2.0228, -2.0228, -2.0228, -2.0228, -2.0228, -2.0228, -2.0228, -2.0228, -2.0228, -2.0228, -2.0228, -2.0228, -2.0228, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.0,0.0, 0.0  , 0.0  , 0.0  ,-2.1160, 0.0, 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  ,-497.50, 0.0   ,  0.0   , 0.0   , 0.0   , 0.0    , 0.0   ,  0.0  }, //HCO interactions with inorganic ions set equal to the parameters of CH2CO groups	  
    {0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.9816, -0.8032, 0.0,0.0,0.0, 0.0  , 0.0   , 0.0   , -0.94280, -0.94280, -0.94280, 0.0, 0.0   , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   , 0.0  }, //CH3COO interactions with inorganic ions missing
    {0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.9816, -0.8032, 0.0,0.0,0.0, 0.0  , 0.0   , 0.0   , -0.94280, -0.94280, -0.94280, 0.0, 0.0   , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   , 0.0  }, //CH2COO interactions with inorganic ions missing
    {0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.8,0.8, 0.0, 0.0  , 0.0  , 0.95080, -1.2588, 0.0   , 0.0   , 0.0   , 0.0  ,-0.96038,-0.96038, 0.0   , 0.0   , 0.0   ,2.6560, 0.0   , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   , 0.0  }, //CH3O interactions with inorganic ions missing
    {0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.8,0.8, 0.0, 0.0  , 0.0  , 0.95080, -1.2588, 0.0   , 0.0   , 0.0   , 0.0  ,-0.96038,-0.96038, 0.0   , 0.0   , 0.0   ,2.6560, 0.0   , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   , 0.0  }, //CH2O interactions with inorganic ions missing
    {0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.8,0.8, 0.0, 0.0  , 0.0  , 0.95080, -1.2588, 0.0   , 0.0   , 0.0   , 0.0  ,-0.96038,-0.96038, 0.0   , 0.0   , 0.0   ,2.6560, 0.0   , 0.0   , 0.0   , 0.0   , 0.0    , 0.0   , 0.0  }, //CHO interactions with inorganic ions missing
    {1.2612, 1.2612, 1.2612, 1.2612, 1.2612, 1.2612, 1.2612, 1.2612, 1.2612, 1.2612, 1.2612, 1.2612, 1.2612, 1.2612, 1.2612, 1.2612, 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,0.8,0.8, 0.0, 0.0  , 0.0  , 0.8, -0.8,0.040284, -1.1912,-1.1912,0.8, 0.0, 0.0, 0.31930, 0.31930, 0.31930, 0.0, 0.0   ,0.0   , 0.0   , 0.0   , 0.0    , 0.0   , 0.0  }, //COOH
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   },  //ACNO2 interactions with numerous groups are missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CHONO2 interactions with several groups and inorganic ions are missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CONO2 interactions with several groups and inorganic ions are missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CH2O-OH interactions with several groups and inorganic ions are missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }, //CHO-OH interactions with several groups and inorganic ions are missing
    {0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,   0.0   ,    0.0   , 0.0   , 0.0    ,  0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0  , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0   ,  0.0 , 0.0   , 0.0   , 0.0   ,  0.0   , 0.0   , 0.0   }}; //CO-OH interactions with several groups and inorganic ions are missing

  //default structure (used if there is no group in the given structure of the molecule)
  double default_structure[45]={0.69,12.74,0.0,0.0, // group C
                                0.0,0.0,0.0,0.0, // group C[OH] 
                                0.0,0.0,0.0,0.0, // group Clacohol
                                0.0,0.0,0.0,0.0, // group Clacohol-tail
                                0.0,0.05,0.0,0.0,0.0, //group C=C
                                1.42,0.86, //group aromatic carbon (AC)
                                0.0,0.15,0.0, // group //AC-C
                                0.0,  //group OH
                                0.0, //group H2O
                                0.15, //group ACOH
                                0.15,0.0, //group ketone
                                0.0,   //group aldehyde  
                                0.0,0.0, //group ester
                                0.30,0.0,0.0, //group ether
                                1.01,  //group acid
                                0.0,   //group ACNO2
                                0.0,0.0,0.0, //group NO3
                                0.0,0.0,0.0}; //group CO-OH

  int i,j,k; 
  int n=surrogate.size();
  // Unifac ions matrixes
  config.nion_unifac=0;
  if (config.SR_ions)
    for (i=0;i<n;i++)
      if (surrogate[i].is_organic==false and surrogate[i].is_inorganic_precursor==false and i!=config.iH2O)
        config.nion_unifac++;
  
  config.RGions.resize(config.nion_unifac);
  config.QGions.resize(config.nion_unifac);
  config.Lions.resize(config.nion_unifac);
  int iion=0;
  if (config.SR_ions)
    for (i=0;i<n;i++)
      if (surrogate[i].is_organic==false and surrogate[i].is_inorganic_precursor==false and i!=config.iH2O)
        {
          config.RGions(iion)=RGions[surrogate[i].index_ion_aiomfac];
          config.QGions(iion)=QGions[surrogate[i].index_ion_aiomfac];
          config.Lions(iion)=config.Z/2*(config.RGions(iion)-config.QGions(iion))-(config.RGions(iion)-1.0);
          iion++;
        }

  //modification of the matrixes to keep only necessary informations:
  //      - remove unused groups in each phase    
  double sum_group[45];
  double sum_group_species;
  int idefault=-1;
  int index=0;
  bool composition_already_defined=true;

  for (j=0;j<NFUNC;++j)
    sum_group[j] = 0.0;
  
  for (i=0;i<n;++i)
    if (surrogate[i].hydrophobic and surrogate[i].compute_gamma_org)
      {
        sum_group_species = 0.0;
        for (j=0;j<NFUNC;++j)
          {
            sum_group_species += surrogate[i].groups[j];
            sum_group[j] += surrogate[i].groups[j];
          }
		
        if (sum_group_species == 0.0)
          {
            if (idefault < 0)
              {
                idefault = i;
                surrogate[i].index_gamma_org = index;
                ++index;
                for (j=0;j<NFUNC;++j)
                  sum_group[j] += default_structure[j];
              }
            else
              surrogate[i].index_gamma_org = surrogate[idefault].index_gamma_org;
          }
        else
          {
            if (i > 0 and index > 0)
              {
                composition_already_defined=false;
                for (k=0;k<i;++k)
                  if (surrogate[k].hydrophobic and surrogate[k].compute_gamma_org
                      and composition_already_defined==false)
                    {
                      composition_already_defined=true;
                      for (j=0;j<NFUNC;++j)
                        if (surrogate[i].groups[j] != surrogate[k].groups[j])
                          composition_already_defined=false;
                      if (composition_already_defined)
                        surrogate[i].index_gamma_org=surrogate[k].index_gamma_org;
                    }
                if (composition_already_defined==false)
                  {
                    surrogate[i].index_gamma_org = index;
                    ++index;
                  }
              }
            else
              {
                surrogate[i].index_gamma_org = index;
                ++index;
              }
          }
	  
      }
    else
      surrogate[i].index_gamma_org = -1;

  config.nfunc_org=0;
  for (j=0;j<NFUNC;++j)
    if (sum_group[j]>0.0)
      config.nfunc_org++;
  
  config.nmol_org=index;

  if (config.nmol_org>0)
    { 
      config.Inter_org.resize(config.nfunc_org,config.nfunc_org);
      config.InterB_org.resize(config.nfunc_org,config.nfunc_org);
      config.InterC_org.resize(config.nfunc_org,config.nfunc_org);
      config.groups_org.resize(config.nfunc_org,config.nmol_org);
      config.RG_org.resize(config.nfunc_org);
      config.QG_org.resize(config.nfunc_org);
	  
      int index_group1=0;
      for (j=0;j<NFUNC;++j)
        if (sum_group[j]>0.0)
          {
            config.RG_org(index_group1)=RG[j];
            config.QG_org(index_group1)=QG[j];
            int index_group2=0;
            for (k=0;k<NFUNC;++k)
              if (sum_group[k]>0.0)
                {
                  config.Inter_org(index_group1,index_group2)=A[j][k];
                  config.InterB_org(index_group1,index_group2)=B[j][k];
		    config.InterC_org(index_group1,index_group2)=C[j][k];
                  index_group2++;
                }
            index_group1++;
          }

      for (i=0;i<n;++i)
        if (surrogate[i].hydrophobic and surrogate[i].compute_gamma_org)
          {
            index_group1=0;
            sum_group_species = 0.0;
            for (j=0;j<NFUNC;++j)
              {
                sum_group_species += surrogate[i].groups[j];
              }
			
			
            for (j=0;j<NFUNC;++j)
              if (sum_group[j]>0.0)
                {
                  if (sum_group_species==0.0)
                    config.groups_org(index_group1,surrogate[i].index_gamma_org)=
                      default_structure[j];
                  else
                    config.groups_org(index_group1,surrogate[i].index_gamma_org)=
                      surrogate[i].groups[j];
                  index_group1++;
                }
          }

      //Pre-calculation of some unifac parameters
      config.Rparam_org.resize(config.nmol_org);
      config.Qparam_org.resize(config.nmol_org);
      config.Lparam_org.resize(config.nmol_org); 
     
      for (i=0;i<config.nmol_org;i++)
        {
          config.Rparam_org(i)=0.0;
          config.Qparam_org(i)=0.0;
          for (j=0;j<config.nfunc_org;++j)
            {
              config.Rparam_org(i)+=config.groups_org(j,i)*config.RG_org(j);
              config.Qparam_org(i)+=config.groups_org(j,i)*config.QG_org(j);
            }
          config.Lparam_org(i)=config.Z/2*(config.Rparam_org(i)-config.Qparam_org(i))-(config.Rparam_org(i)-1.0);
        }
	  
	  
    }


  index=0;
  idefault=-1;
  
  for (j=0;j<NFUNC;++j)
    sum_group[j] = 0.0;
  
  for (i=0;i<n;++i)
    if (surrogate[i].hydrophilic and surrogate[i].compute_gamma_aq)
      {
        sum_group_species = 0.0;
        for (j=0;j<NFUNC;++j)
          {
            sum_group_species += surrogate[i].groups[j];
            sum_group[j] += surrogate[i].groups[j];
          }
		
        if (sum_group_species == 0.0)
          {
            if (idefault < 0)
              {
                idefault = i;
                surrogate[i].index_gamma_aq = index;
                ++index;
                for (j=0;j<NFUNC;++j)
                  sum_group[j] += default_structure[j];
              }
            else
              surrogate[i].index_gamma_aq = surrogate[idefault].index_gamma_aq;
          }
        else
          {
            if (i > 0 and index > 0)
              {
                composition_already_defined=false;
                for (k=0;k<i;++k)
                  if (surrogate[k].hydrophilic and surrogate[k].compute_gamma_aq
                      and composition_already_defined==false)
                    {
                      composition_already_defined=true;
                      for (j=0;j<NFUNC;++j)
                        if (surrogate[i].groups[j] != surrogate[k].groups[j])
                          composition_already_defined=false;
                      if (composition_already_defined)
                        surrogate[i].index_gamma_aq=surrogate[k].index_gamma_aq;
                    }
                if (composition_already_defined==false)
                  {
                    surrogate[i].index_gamma_aq = index;
                    ++index;
                  }
              }
            else
              {
                surrogate[i].index_gamma_aq = index;
                ++index;
              }
          }
	  
      }
    else
      surrogate[i].index_gamma_aq = -1;

  config.nfunc_aq=0;
  for (j=0;j<NFUNC;++j)
    if (sum_group[j]>0.0)
      ++config.nfunc_aq;
  
  config.nmol_aq=index;

  if (config.nmol_aq>0)
    {
      config.Inter_aq.resize(config.nfunc_aq,config.nfunc_aq);
      config.InterB_aq.resize(config.nfunc_aq,config.nfunc_aq);
      config.InterC_aq.resize(config.nfunc_aq,config.nfunc_aq);
      config.groups_aq.resize(config.nfunc_aq,config.nmol_aq);
      config.RG_aq.resize(config.nfunc_aq);
      config.QG_aq.resize(config.nfunc_aq);
	  
      int index_group1=0;
      for (j=0;j<NFUNC;++j)
        if (sum_group[j]>0.0)
          {
            config.RG_aq(index_group1)=RG[j];
            config.QG_aq(index_group1)=QG[j];
            int index_group2=0;
            for (k=0;k<NFUNC;++k)
              if (sum_group[k]>0.0)
                {
                  config.Inter_aq(index_group1,index_group2)=A[j][k];
		  config.InterB_aq(index_group1,index_group2)=B[j][k];
		  config.InterC_aq(index_group1,index_group2)=C[j][k];
                  ++index_group2;
                }
            ++index_group1;
          }

      for (i=0;i<n;++i)
        if (surrogate[i].hydrophilic and surrogate[i].compute_gamma_aq)
          {
            index_group1=0;
            sum_group_species = 0.0;
            for (j=0;j<NFUNC;++j)
              {
                sum_group_species += surrogate[i].groups[j];
              }
			
			
            for (j=0;j<NFUNC;++j)
              if (sum_group[j]>0.0)
                {
                  if (sum_group_species==0.0)
                    config.groups_aq(index_group1,surrogate[i].index_gamma_aq)=
                      default_structure[j];
                  else
                    config.groups_aq(index_group1,surrogate[i].index_gamma_aq)=
                      surrogate[i].groups[j];
                  index_group1++;
                }
          }	 

      //Pre-calculation of some unifac parameters
      config.Rparam_aq.resize(config.nmol_aq);
      config.Qparam_aq.resize(config.nmol_aq);
      config.Lparam_aq.resize(config.nmol_aq);      
      for (i=0;i<config.nmol_aq;i++)
        {
          config.Rparam_aq(i)=0.0;
          config.Qparam_aq(i)=0.0;
          for (j=0;j<config.nfunc_aq;++j)
            {
              config.Rparam_aq(i)+=config.groups_aq(j,i)*config.RG_aq(j);
              config.Qparam_aq(i)+=config.groups_aq(j,i)*config.QG_aq(j);
            }
          config.Lparam_aq(i)=config.Z/2*(config.Rparam_aq(i)-config.Qparam_aq(i))-(config.Rparam_aq(i)-1.0);
        }
	  
 
    }

  

  index=0;
  idefault=-1;
  
  for (j=0;j<NFUNC;++j)
    sum_group[j] = 0.0;
  
  for (i=0;i<n;++i)
    if (surrogate[i].compute_gamma_org)
      {
        sum_group_species = 0.0;
        for (j=0;j<NFUNC;++j)
          {
            sum_group_species += surrogate[i].groups[j];
            sum_group[j] += surrogate[i].groups[j];
          }
		
        if (sum_group_species == 0.0)
          {
            if (idefault < 0)
              {
                idefault = i;
                surrogate[i].index_gamma_tot = index;
                ++index;
                for (j=0;j<NFUNC;++j)
                  sum_group[j] += default_structure[j];
              }
            else
              surrogate[i].index_gamma_tot = surrogate[idefault].index_gamma_tot;
          }
        else
          {
            if (i > 0 and index > 0)
              {
                composition_already_defined=false;
                for (k=0;k<i;++k)
                  if (surrogate[k].compute_gamma_org and composition_already_defined==false)
                    {
                      composition_already_defined=true;
                      for (j=0;j<NFUNC;++j)
                        if (surrogate[i].groups[j] != surrogate[k].groups[j])
                          composition_already_defined=false;
                      if (composition_already_defined)
                        surrogate[i].index_gamma_tot=surrogate[k].index_gamma_tot;
                    }
                if (composition_already_defined==false)
                  {
                    surrogate[i].index_gamma_tot = index;
                    ++index;
                  }
              }
            else
              {
                surrogate[i].index_gamma_tot = index;
                ++index;
              }
          }
	  
      }
    else
      surrogate[i].index_gamma_tot = -1;

  config.nfunc_tot=0;
  for (j=0;j<NFUNC;++j)
    if (sum_group[j]>0.0)
      ++config.nfunc_tot;
  
  config.nmol_tot=index;

  if (config.nmol_tot>0)
    {
      config.Inter_tot.resize(config.nfunc_tot,config.nfunc_tot);
      config.InterB_tot.resize(config.nfunc_tot,config.nfunc_tot);
      config.InterC_tot.resize(config.nfunc_tot,config.nfunc_tot);
      config.groups_tot.resize(config.nfunc_tot,config.nmol_tot);
      config.RG_tot.resize(config.nfunc_tot);
      config.QG_tot.resize(config.nfunc_tot);
	  
      int index_group1=0;
      for (j=0;j<NFUNC;++j)
        if (sum_group[j]>0.0)
          {
            config.RG_tot(index_group1)=RG[j];
            config.QG_tot(index_group1)=QG[j];
            int index_group2=0;
            for (k=0;k<NFUNC;++k)
              if (sum_group[k]>0.0)
                {
                  config.Inter_tot(index_group1,index_group2)=A[j][k];
		  config.InterB_tot(index_group1,index_group2)=B[j][k];
		  config.InterC_tot(index_group1,index_group2)=C[j][k];
                  ++index_group2;
                }
            ++index_group1;
          }

      for (i=0;i<n;++i)
        if (surrogate[i].compute_gamma_org)
          {
            index_group1=0;
            sum_group_species = 0.0;
            for (j=0;j<NFUNC;++j)
              {
                sum_group_species += surrogate[i].groups[j];
              }
			
			
            for (j=0;j<NFUNC;++j)
              if (sum_group[j]>0.0)
                {
                  if (sum_group_species==0.0)
                    config.groups_tot(index_group1,surrogate[i].index_gamma_tot)=
                      default_structure[j];
                  else
                    config.groups_tot(index_group1,surrogate[i].index_gamma_tot)=
                      surrogate[i].groups[j];
                  ++index_group1;
                }
          }

      //Pre-calculation of some unifac parameters
      config.Rparam_tot.resize(config.nmol_tot);
      config.Qparam_tot.resize(config.nmol_tot);
      config.Lparam_tot.resize(config.nmol_tot);
      
      for (i=0;i<config.nmol_tot;i++)
        {
          config.Rparam_tot(i)=0.0;
          config.Qparam_tot(i)=0.0;
          for (j=0;j<config.nfunc_tot;++j)
            {
              config.Rparam_tot(i)+=config.groups_tot(j,i)*config.RG_tot(j);
              config.Qparam_tot(i)+=config.groups_tot(j,i)*config.QG_tot(j);
            }
          config.Lparam_tot(i)=config.Z/2*(config.Rparam_tot(i)-config.Qparam_tot(i))-(config.Rparam_tot(i)-1.0);
        }
	  
    }

  if (config.tabulation_unifac)
    {
      FILE *f;
      string file_out="unifac.csv";
      f=fopen(file_out.c_str(),"wb");  
      fprintf (f, "%i %s", config.iH2O, "\n");
      for (i=0;i<=config.iH2O;i++)
        {
          int a=0;
          int b=0;
          if (surrogate[i].hydrophilic)
            a=1;
          if (surrogate[i].hydrophobic)
            b=1;
          fprintf (f, "%i %s %i %s %i", i, ";", a, ";", b); 
          for (j=0;j<NFUNC;j++)
            fprintf (f, "%s %e", ";", surrogate[i].groups[j]); 
          fprintf (f, "%s", "\n");
        }

      for (i=0;i<=config.iH2O;i++)
        fprintf (f, "%i %s %i %s %i %s", surrogate[i].index_gamma_tot, ";", surrogate[i].index_gamma_org, ";", surrogate[i].index_gamma_aq, "\n");         

      fprintf (f, "%i %s %i %s", config.nmol_tot, ";", config.nfunc_tot, "\n");
      for (j=0;j<config.nfunc_tot;j++)
        for (k=0;k<config.nfunc_tot;k++)      
          fprintf (f, "%i %s %i %s %e %s %e %s %e %s", j, ";", k, ";", config.Inter_tot(j,k), ";", config.InterB_tot(j,k), ";", config.InterC_tot(j,k),"\n");        
     
      for (i=0;i<config.nmol_tot;i++)    
        for (k=0;k<config.nfunc_tot;k++)      
          fprintf (f, "%i %s %i %s %e %s", k, ";", i, ";", config.groups_tot(k,i), "\n");    

      for (j=0;j<config.nfunc_tot;j++)
        fprintf(f, "%i %s %e %s %e %s", j, ";", config.RG_tot(j), ";", config.QG_tot(j), "\n");

      for (i=0;i<config.nmol_tot;i++)  
        fprintf(f, "%i %s %e %s %e %s %e %s", i, ";", config.Rparam_tot(i), ";", config.Qparam_tot(i), ";", config.Lparam_tot(i), "\n");


      fprintf (f, "%i %s %i %s", config.nmol_org, ";", config.nfunc_org, "\n");
      for (j=0;j<config.nfunc_org;j++)
        for (k=0;k<config.nfunc_org;k++)      
          fprintf (f, "%i %s %i %s %e %s %e %s %e %s", j, ";", k, ";", config.Inter_org(j,k),  ";", config.InterB_org(j,k),  ";", config.InterC_org(j,k),"\n");        
     
      for (i=0;i<config.nmol_org;i++)    
        for (k=0;k<config.nfunc_org;k++)      
          fprintf (f, "%i %s %i %s %e %s", k, ";", i, ";", config.groups_org(k,i), "\n");    

      for (j=0;j<config.nfunc_org;j++)
        fprintf(f, "%i %s %e %s %e %s", j, ";", config.RG_org(j), ";", config.QG_org(j), "\n");

      for (i=0;i<config.nmol_org;i++)  
        fprintf(f, "%i %s %e %s %e %s %e %s", i, ";", config.Rparam_org(i), ";", config.Qparam_org(i), ";", config.Lparam_org(i), "\n");


      fprintf (f, "%i %s %i %s", config.nmol_aq, ";", config.nfunc_aq, "\n");
      for (j=0;j<config.nfunc_aq;j++)
        for (k=0;k<config.nfunc_aq;k++)      
          fprintf (f, "%i %s %i %s %e %s %e %s %e %s", j, ";", k, ";", config.Inter_aq(j,k), ";", config.InterB_aq(j,k),";", config.InterC_aq(j,k),  "\n");        
     
      for (i=0;i<config.nmol_aq;i++)    
        for (k=0;k<config.nfunc_aq;k++)      
          fprintf (f, "%i %s %i %s %e %s", k, ";", i, ";", config.groups_aq(k,i), "\n");    

      for (j=0;j<config.nfunc_aq;j++)
        fprintf(f, "%i %s %e %s %e %s", j, ";", config.RG_aq(j), ";", config.QG_aq(j), "\n");

      for (i=0;i<config.nmol_aq;i++)  
        fprintf(f, "%i %s %e %s %e %s %e %s", i, ";", config.Rparam_aq(i), ";", config.Qparam_aq(i), ";", config.Lparam_aq(i), "\n");


      config.ntemp=60;
      for (i=0;i<n;i++)
        surrogate[i].species_activity.resize(config.ntemp);

      int itemp;
      double sum_group2;
      for (i=0;i<n;i++)
        if (surrogate[i].is_organic or i==config.iH2O)
          {
            sum_group2=0.0;
            for (j=0;j<NFUNC;j++)
              if (surrogate[i].groups[j]>0.0)
                sum_group2=1;

            if (sum_group2==0.0)
              for (j=0;j<NFUNC;j++)
                surrogate[i].groups[j]=default_structure[j];
          }

      for (itemp=0;itemp<config.ntemp;itemp++)
        {     
          double Temperature=263.15+1.0*itemp;
          Array<double,2> Inter2;
          Inter2.resize(NFUNC,NFUNC);
          Array <double, 2> group_activity_mol,sum2mol;
          group_activity_mol.resize(NFUNC,n);
          sum2mol.resize(NFUNC,n);
          Array <double, 2> surface_fraction_mol;      
          surface_fraction_mol.resize(NFUNC,n);      
          surface_fraction_mol=0.0;
          double sum_surf=0.0;
          for (i=0;i<n;i++)
            if (surrogate[i].is_organic or surrogate[i].name=="H2O")
              {
                double sum_surf_mol=0.0;
                for (j=0;j<NFUNC;j++)
                  if (surrogate[i].groups[j]>0.0)
                    {               
                      surface_fraction_mol(j,i)+=QG[j]*surrogate[i].groups[j];               
                      sum_surf_mol+=QG[j]*surrogate[i].groups[j];  
                    }

                for (j=0;j<NFUNC;j++)      
                  surface_fraction_mol(j,i)/=sum_surf_mol;
              }

          for (j=0;j<NFUNC;j++)
            for (k=0;k<NFUNC;k++)          
              Inter2(j,k)=exp(-A[j][k]/Temperature+B[j][k]*(1.0/298.15-1.0/298)+C[j][k]*(298.15/Temperature-1+log(Temperature/298.15)));                  

          for (i=0;i<n;i++)
            if (surrogate[i].is_organic or surrogate[i].name=="H2O")
              for (j=0;j<NFUNC;j++)
                if (surrogate[i].groups[j]>0.0)
                  {
                    sum2mol(j,i)=0.0;
                    for (k=0;k<NFUNC;k++)
                      if (surrogate[i].groups[k]>0.0)
                        sum2mol(j,i)+=surface_fraction_mol(k,i)*Inter2(k,j);              
                  }

  
          for (i=0;i<n;i++)
            if (surrogate[i].is_organic or surrogate[i].name=="H2O")
              {
                surrogate[i].species_activity(itemp)=0.0;
                for (j=0;j<NFUNC;j++)
                  if (surrogate[i].groups[j]>0.0)
                    {
                      group_activity_mol(j,i)=1.0-log(sum2mol(j,i));                     
                      for (k=0;k<NFUNC;k++)      
                        if (surrogate[i].groups[k]>0.0)
                          group_activity_mol(j,i)-=surface_fraction_mol(k,i)*Inter2(j,k)/sum2mol(k,i);  

                      surrogate[i].species_activity(itemp)+=surrogate[i].groups[j]*QG[j]*group_activity_mol(j,i);

                    }     
                 fprintf (f, "%i %s %i %s %e %s", itemp, ";", i, ";", surrogate[i].species_activity(itemp), "\n");
              }           
        }
 
      fclose(f);
    }  
}


void read_tab_unifac(model_config &config, vector<species>& surrogate)
{
  FILE *f;
  string file_out="unifac.csv";  
  ifstream fichier(file_out.c_str(), ios::in);
  string ligne;
  int NFUNC=45;
  int okfile=1;
  int n=surrogate.size();
  getline(fichier, ligne, '\n');
  int ih2oin=atoi(ligne.c_str()); 
  if (ih2oin!=config.iH2O)
    {
      okfile=0;
      cout << "ih2o: " << ih2oin << " " << config.iH2O << endl;
    }
  int i,j,k,itemp;
  config.Z=10.0;
  
  for (i=0;i<=config.iH2O;i++)
    {
      getline(fichier, ligne, ';');
      int ispecies=atoi(ligne.c_str()); 
      getline(fichier, ligne, ';');
      int a=atoi(ligne.c_str()); 
      getline(fichier, ligne, ';');
      int b=atoi(ligne.c_str());            
      if ((surrogate[i].hydrophilic and a!=1) or (surrogate[i].hydrophilic==false and a==1))
        {
          okfile=0;
          cout << surrogate[i].name << " hydrophilic problem: " << surrogate[i].hydrophilic << " " << a << endl;
        }
      if ((surrogate[i].hydrophobic and b!=1) or (surrogate[i].hydrophobic==false and b==1))
        {
          okfile=0;
          cout << surrogate[i].name << " hydrophobic problem: " << surrogate[i].hydrophobic << " " << b << endl;
        }
      
      for (j=0;j<NFUNC-1;j++)
        {
          getline(fichier, ligne, ';');
          double g=atof(ligne.c_str());
          if (g!=surrogate[i].groups[j])
            okfile=0;
        }
      j=NFUNC-1;
      getline(fichier, ligne, '\n');
      double g=atof(ligne.c_str());
      if (g!=surrogate[i].groups[j])
        okfile=0;
    }  
 
  if (okfile==0)
    {
      fichier.close();
      param_unifac(config, surrogate);
    }
  else
    {
      for (i=0;i<=config.iH2O;i++)
        {
          getline(fichier, ligne, ';');
          surrogate[i].index_gamma_tot=atoi(ligne.c_str()); 
          getline(fichier, ligne, ';');
          surrogate[i].index_gamma_org=atoi(ligne.c_str()); 
          getline(fichier, ligne, '\n');
          surrogate[i].index_gamma_aq=atoi(ligne.c_str()); 
        }

      for (i=config.iH2O+1;i<n;i++)
        {
          surrogate[i].index_gamma_tot=-1;           
          surrogate[i].index_gamma_org=-1;           
          surrogate[i].index_gamma_aq=-1; 
        }

      getline(fichier, ligne, ';');
      config.nmol_tot=atoi(ligne.c_str()); 
      getline(fichier, ligne, '\n');
      config.nfunc_tot=atoi(ligne.c_str()); 
      config.Inter_tot.resize(config.nfunc_tot,config.nfunc_tot);            
      config.InterB_tot.resize(config.nfunc_tot,config.nfunc_tot); 
      config.InterC_tot.resize(config.nfunc_tot,config.nfunc_tot); 
      config.groups_tot.resize(config.nfunc_tot,config.nmol_tot);
      config.RG_tot.resize(config.nfunc_tot);
      config.QG_tot.resize(config.nfunc_tot);
      config.Rparam_tot.resize(config.nmol_tot);
      config.Qparam_tot.resize(config.nmol_tot);
      config.Lparam_tot.resize(config.nmol_tot);

      for (j=0;j<config.nfunc_tot;j++)
        for (k=0;k<config.nfunc_tot;k++)      
          {
            getline(fichier, ligne, ';');
            int a=atoi(ligne.c_str()); 
            getline(fichier, ligne, ';');
            int b=atoi(ligne.c_str()); 
            getline(fichier, ligne, ';');
            config.Inter_tot(j,k)=atof(ligne.c_str());             
	    getline(fichier, ligne, ';');
            config.InterB_tot(j,k)=atof(ligne.c_str());
	    getline(fichier, ligne, '\n');
            config.InterC_tot(j,k)=atof(ligne.c_str());
          }        
  
      for (i=0;i<config.nmol_tot;i++)    
        for (k=0;k<config.nfunc_tot;k++)      
          {
            getline(fichier, ligne, ';');
            int a=atoi(ligne.c_str()); 
            getline(fichier, ligne, ';');
            int b=atoi(ligne.c_str()); 
            getline(fichier, ligne, '\n');
            config.groups_tot(k,i)=atof(ligne.c_str());  
          }         

  
      for (j=0;j<config.nfunc_tot;j++)
        {
          getline(fichier, ligne, ';');
          int a=atoi(ligne.c_str()); 
          getline(fichier, ligne, ';');
          config.RG_tot(j)=atof(ligne.c_str()); 
          getline(fichier, ligne, '\n');
          config.QG_tot(j)=atof(ligne.c_str()); 
        }
 
      for (i=0;i<config.nmol_tot;i++)  
        {
          getline(fichier, ligne, ';');
          int a=atoi(ligne.c_str()); 
          getline(fichier, ligne, ';');
          config.Rparam_tot(i)=atof(ligne.c_str()); 
          getline(fichier, ligne, ';');
          config.Qparam_tot(i)=atof(ligne.c_str()); 
          getline(fichier, ligne, '\n');
          config.Lparam_tot(i)=atof(ligne.c_str()); 
        }
 
      getline(fichier, ligne, ';');
      config.nmol_org=atoi(ligne.c_str()); 
      getline(fichier, ligne, '\n');
      config.nfunc_org=atoi(ligne.c_str()); 
      config.Inter_org.resize(config.nfunc_org,config.nfunc_org); 
      config.InterB_org.resize(config.nfunc_org,config.nfunc_org); 
      config.InterC_org.resize(config.nfunc_org,config.nfunc_org); 
      config.groups_org.resize(config.nfunc_org,config.nmol_org);
      config.RG_org.resize(config.nfunc_org);
      config.QG_org.resize(config.nfunc_org);
      config.Rparam_org.resize(config.nmol_org);
      config.Qparam_org.resize(config.nmol_org);
      config.Lparam_org.resize(config.nmol_org);

      for (j=0;j<config.nfunc_org;j++)
        for (k=0;k<config.nfunc_org;k++)      
          {
            getline(fichier, ligne, ';');
            int a=atoi(ligne.c_str()); 
            getline(fichier, ligne, ';');
            int b=atoi(ligne.c_str()); 
            getline(fichier, ligne, ';');
            config.Inter_org(j,k)=atof(ligne.c_str());             
	    getline(fichier, ligne, ';');
	    config.InterB_org(j,k)=atof(ligne.c_str());  
	    getline(fichier, ligne, '\n');
            config.InterC_org(j,k)=atof(ligne.c_str());  
          }        
  
      for (i=0;i<config.nmol_org;i++)    
        for (k=0;k<config.nfunc_org;k++)      
          {
            getline(fichier, ligne, ';');
            int a=atoi(ligne.c_str()); 
            getline(fichier, ligne, ';');
            int b=atoi(ligne.c_str()); 
            getline(fichier, ligne, '\n');
            config.groups_org(k,i)=atof(ligne.c_str());  
          }         

  
      for (j=0;j<config.nfunc_org;j++)
        {
          getline(fichier, ligne, ';');
          int a=atoi(ligne.c_str()); 
          getline(fichier, ligne, ';');
          config.RG_org(j)=atof(ligne.c_str()); 
          getline(fichier, ligne, '\n');
          config.QG_org(j)=atof(ligne.c_str()); 
        }
 
      for (i=0;i<config.nmol_org;i++)  
        {
          getline(fichier, ligne, ';');
          int a=atoi(ligne.c_str()); 
          getline(fichier, ligne, ';');
          config.Rparam_org(i)=atof(ligne.c_str()); 
          getline(fichier, ligne, ';');
          config.Qparam_org(i)=atof(ligne.c_str()); 
          getline(fichier, ligne, '\n');
          config.Lparam_org(i)=atof(ligne.c_str()); 
        }

      getline(fichier, ligne, ';');
      config.nmol_aq=atoi(ligne.c_str()); 
      getline(fichier, ligne, '\n');
      config.nfunc_aq=atoi(ligne.c_str()); 
      config.Inter_aq.resize(config.nfunc_aq,config.nfunc_aq);            
      config.InterB_aq.resize(config.nfunc_aq,config.nfunc_aq);  
      config.InterC_aq.resize(config.nfunc_aq,config.nfunc_aq);  
      config.groups_aq.resize(config.nfunc_aq,config.nmol_aq);
      config.RG_aq.resize(config.nfunc_aq);
      config.QG_aq.resize(config.nfunc_aq);
      config.Rparam_aq.resize(config.nmol_aq);
      config.Qparam_aq.resize(config.nmol_aq);
      config.Lparam_aq.resize(config.nmol_aq);

      for (j=0;j<config.nfunc_aq;j++)
        for (k=0;k<config.nfunc_aq;k++)      
          {
            getline(fichier, ligne, ';');
            int a=atoi(ligne.c_str()); 
            getline(fichier, ligne, ';');
            int b=atoi(ligne.c_str()); 
            getline(fichier, ligne, ';');
            config.Inter_aq(j,k)=atof(ligne.c_str());             
            getline(fichier, ligne, ';');
            config.InterB_aq(j,k)=atof(ligne.c_str());  
            getline(fichier, ligne, '\n');
            config.InterC_aq(j,k)=atof(ligne.c_str());  
          }        
  
      for (i=0;i<config.nmol_aq;i++)    
        for (k=0;k<config.nfunc_aq;k++)      
          {
            getline(fichier, ligne, ';');
            int a=atoi(ligne.c_str()); 
            getline(fichier, ligne, ';');
            int b=atoi(ligne.c_str()); 
            getline(fichier, ligne, '\n');
            config.groups_aq(k,i)=atof(ligne.c_str());  
          }         

  
      for (j=0;j<config.nfunc_aq;j++)
        {
          getline(fichier, ligne, ';');
          int a=atoi(ligne.c_str()); 
          getline(fichier, ligne, ';');
          config.RG_aq(j)=atof(ligne.c_str()); 
          getline(fichier, ligne, '\n');
          config.QG_aq(j)=atof(ligne.c_str()); 
        }
 
      for (i=0;i<config.nmol_aq;i++)  
        {
          getline(fichier, ligne, ';');
          int a=atoi(ligne.c_str()); 
          getline(fichier, ligne, ';');
          config.Rparam_aq(i)=atof(ligne.c_str()); 
          getline(fichier, ligne, ';');
          config.Qparam_aq(i)=atof(ligne.c_str()); 
          getline(fichier, ligne, '\n');
          config.Lparam_aq(i)=atof(ligne.c_str()); 
        }

      config.ntemp=60;
      for (i=0;i<n;i++)
        surrogate[i].species_activity.resize(config.ntemp);

      for (itemp=0;itemp<config.ntemp;itemp++)
        {       
          for (i=0;i<=config.iH2O;i++)           
            {
              getline(fichier, ligne, ';');
              int a=atoi(ligne.c_str()); 
              getline(fichier, ligne, ';');
              int b=atoi(ligne.c_str()); 
              getline(fichier, ligne, '\n');
              surrogate[i].species_activity(itemp)=atof(ligne.c_str());   
            }           
        }

      fichier.close();
    }

  //                    H+    Li+   Na+   K+   NH4+  Mg2+  Ca2+  Cl-   Br-   NO3-  HSO4-  SO4--
  double RGions[12] = {1.78, 0.61, 0.38, 0.44, 0.69, 5.44, 2.24, 0.99, 1.25, 0.95, 1.65,  3.34};
  double QGions[12] = {2.70, 0.99, 0.62, 0.58, 0.78, 8.35, 3.40, 0.99, 1.16, 0.97, 1.40,  3.96};   

  config.nion_unifac=0;
  if (config.SR_ions)
    for (i=0;i<n;i++)
      if (surrogate[i].is_organic==false and surrogate[i].is_inorganic_precursor==false and i!=config.iH2O)
        config.nion_unifac++;
  
  config.RGions.resize(config.nion_unifac);
  config.QGions.resize(config.nion_unifac);
  config.Lions.resize(config.nion_unifac);
  int iion=0;
  if (config.SR_ions)
    for (i=0;i<n;i++)
      if (surrogate[i].is_organic==false and surrogate[i].is_inorganic_precursor==false and i!=config.iH2O)
        {
          config.RGions(iion)=RGions[surrogate[i].index_ion_aiomfac];
          config.QGions(iion)=QGions[surrogate[i].index_ion_aiomfac];
          config.Lions(iion)=config.Z/2*(config.RGions(iion)-config.QGions(iion))-(config.RGions(iion)-1.0);
          iion++;
        }



}


void check_config(model_config &config, vector<species>& surrogate)
{
  // Chech if there is problem in the used parameters of species.cxx
  int n=surrogate.size();
  int i;
  double tiny=1.0e-100;
  bool badly_formatted=false;
  
  for (i=0;i<n;++i)
    if(surrogate[i].MM<=tiny)
      {
        cout << "WARNING: bad input for MM of species " << surrogate[i].name << endl;
        badly_formatted=true;
      }
  
  for (i=0;i<n;++i)
    if (surrogate[i].is_organic)
      if(surrogate[i].nonvolatile==false) 
        {
          if (surrogate[i].Tref<=tiny)
            {
              cout << "WARNING: bad input for Tref of species " << surrogate[i].name << endl;
              badly_formatted=true;
            }
          if (surrogate[i].deltaH<=tiny)
            {
              cout << "WARNING: bad input for deltaH of species " << surrogate[i].name << endl;
              badly_formatted=true;
            }
		
          if (surrogate[i].hydrophilic)
            {
              if (surrogate[i].compute_gamma_aq and surrogate[i].Henry<=tiny and
                  surrogate[i].Psat_ref>=tiny)
                cout << "Henry's law constant of species " << surrogate[i].name
                     << " is computed from the saturation vapor pressure" << endl;
              else if (surrogate[i].compute_gamma_aq and surrogate[i].Henry<=tiny and
                       surrogate[i].Psat_ref<=tiny)
                {
                  cout << "WARNING: bad input for Psat_ref or Henry of species "
                       << surrogate[i].name << endl;
                  badly_formatted=true;
                }
              else if (surrogate[i].compute_gamma_aq==false and surrogate[i].Henry<=tiny)
                {
                  cout << "WARNING: bad input for Henry of species test "
                       << surrogate[i].name << endl;
                  badly_formatted=true;
                }
              if (surrogate[i].aq_type!="diacid" and surrogate[i].aq_type!="monoacid"
                  and surrogate[i].aq_type!="aldehyde" and surrogate[i].aq_type!="none")
                {
                  cout << "WARNING: aq_type " << surrogate[i].aq_type << " of species "
                       << surrogate[i].name << " not defined." << endl;
                  badly_formatted=true;
                }
              if (surrogate[i].aq_type=="diacid" and surrogate[i].Kacidity2<=tiny)
                {
                  cout << "WARNING: bad input for Kacidity2 of diacid species "
                       << surrogate[i].name << endl;
                  badly_formatted=true;
                }
              if ((surrogate[i].aq_type=="diacid" or surrogate[i].aq_type=="monoacid")
                  and surrogate[i].Kacidity1<=tiny)
                {
                  cout << "WARNING: bad input for Kacidity1 of " << surrogate[i].aq_type
                       << "species " << surrogate[i].name << endl;
                  badly_formatted=true;
                }
              if (surrogate[i].aq_type=="aldehyde" and surrogate[i].pHref<=tiny)
                {
                  cout << "WARNING: bad input for pHref of aldehyde species "
                       << surrogate[i].name << endl;
                  badly_formatted=true;
                }
              if (surrogate[i].aq_type=="aldehyde" and surrogate[i].beta<=tiny)
                {
                  cout << "WARNING: bad input for beta of aldehyde species "
                       << surrogate[i].name << endl;
                  badly_formatted=true;
                }
              if (surrogate[i].aq_type=="aldehyde" and surrogate[i].Koligo_aq<=tiny)
                {
                  cout << "WARNING: bad input for Koligo_aq of aldehyde species "
                       << surrogate[i].name << endl;
                  badly_formatted=true;
                }
            }
		
          if (surrogate[i].kp_from_experiment)
            {
              if (surrogate[i].kp_experiment<=tiny)
                {
                  cout << "WARNING: bad input for kp_experiment of species "
                       << surrogate[i].name << endl;
                  badly_formatted=true;
                }	
            }
          else
            {
              if (surrogate[i].Psat_ref<=tiny)
                {
                  cout << "WARNING: bad input for Psat_ref of species "
                       << surrogate[i].name << surrogate[i].Psat_ref << endl;
                  badly_formatted=true;
                }
			  
              if (surrogate[i].Koligo_org<0.0)
                {
                  cout << "WARNING: bad input for Koligo_org of species "
                       << surrogate[i].name << endl;
                  badly_formatted=true;
                }
            }
		  
        }
  if(badly_formatted)
    exit(0);
  
}

void init_transfert_parameters(model_config &config, vector<species>& surrogate)
{
  int b,ilayer,iphase;
  int n=surrogate.size();
  int i;
  config.nphase.resize(config.nbins,config.nlayer);
  config.Vlayer.resize(config.nlayer);
  config.alpha_layer.resize(config.nlayer);
  config.Alayer.resize(config.nlayer,4);
  if (config.explicit_representation)
    {
      config.dbound.resize(config.nbins,config.nlayer+1);
      config.Radius.resize(config.nbins,config.nlayer);
      for (i=0;i<n;++i)
	surrogate[i].dif_org.resize(config.nbins,config.nlayer);	

      double sum_Vlayer=0.0;
      if (config.explicit_method==1)
	for (ilayer=0;ilayer<config.nlayer;ilayer++)
	  {
	    config.Vlayer(ilayer)=pow((ilayer+1)*1.0/config.nlayer,3.0)-sum_Vlayer;
	    sum_Vlayer+=config.Vlayer(ilayer);
	  }
      else if (config.explicit_method==2)
	for (ilayer=0;ilayer<config.nlayer;ilayer++)
	  config.Vlayer(ilayer)=1.0/config.nlayer;
      else if (config.explicit_method==3)
	{
	  config.Vlayer(0)=1.0;
	  double sumVlayer=1.0;
	  double Red=1.01;
	  for (ilayer=1;ilayer<config.nlayer;ilayer++)
	    { 
	      config.Vlayer(ilayer)=config.Vlayer(ilayer-1)/Red;
	      sumVlayer+=config.Vlayer(ilayer);
	    }
	  for (ilayer=0;ilayer<config.nlayer;ilayer++)
	    config.Vlayer(ilayer)=config.Vlayer(ilayer)/sumVlayer;
	}
   
    }
  else
    {
      if(config.nlayer==1)
	{
	  config.Vlayer(0)=1.0;
	  config.alpha_layer(0)=1.0e20;
	  config.Alayer(0,0)=0.744123644366;
	  config.Alayer(0,1)=-2.42956182268;
	  config.Alayer(0,2)=3.67009239335;
	  config.Alayer(0,3)=-2.98465421503;
	}
      else if(config.nlayer==2)
	{
	  config.Vlayer(0)=0.99;
          config.Vlayer(1)=0.01;
	  config.alpha_layer(0)=1.0;
          config.alpha_layer(1)=1.0e20;
	  config.Alayer(0,0)=0.744123644366;
	  config.Alayer(0,1)=-2.42956182268;
	  config.Alayer(0,2)=3.67009239335;
	  config.Alayer(0,3)=-2.98465421503;
	  config.Alayer(1,0)=0.744123644366;
	  config.Alayer(1,1)=-2.42956182268;
	  config.Alayer(1,2)=3.67009239335;
	  config.Alayer(1,3)=-2.98465421503;
	}
      else if (config.nlayer==3)
	{
	  config.Vlayer(0)=0.6;
	  config.Vlayer(1)=0.39;
	  config.Vlayer(2)=0.01;
	  config.alpha_layer(0)=0.98776;
	  config.alpha_layer(1)=6.25580;
	  config.alpha_layer(2)=1.0e20;
	  config.Alayer(0,0)=0.744123644366;
	  config.Alayer(0,1)=-2.42956182268;
	  config.Alayer(0,2)=3.67009239335;
	  config.Alayer(0,3)=-2.98465421503;
	  config.Alayer(1,0)=-0.207952538556;
	  config.Alayer(1,1)=0.861371078474;
	  config.Alayer(1,2)=-0.109866578569;
	  config.Alayer(1,3)=-1.54355196135;
	  config.Alayer(2,0)=0.833327936496;
	  config.Alayer(2,1)=-2.25743854559;
	  config.Alayer(2,2)=3.06448590934;
	  config.Alayer(2,3)=-2.64037530024;
	}
      else if (config.nlayer==4)
	{
	  config.Vlayer(0)=0.6;
	  config.Vlayer(1)=0.26;
	  config.Vlayer(2)=0.13;
	  config.Vlayer(3)=0.01;
	  config.alpha_layer(0)=0.98776;
	  config.alpha_layer(1)=6.25580;
	  config.alpha_layer(2)=68.8666;
	  config.alpha_layer(3)=1.0e20;
	  config.Alayer(0,0)=0.744123644366;
	  config.Alayer(0,1)=-2.42956182268;
	  config.Alayer(0,2)=3.67009239335;
	  config.Alayer(0,3)=-2.98465421503;
	  config.Alayer(1,0)=-0.207952538556;
	  config.Alayer(1,1)=0.861371078474;
	  config.Alayer(1,2)=-0.109866578569;
	  config.Alayer(1,3)=-1.54355196135;
	  config.Alayer(2,0)=0.833327936496;
	  config.Alayer(2,1)=-2.25743854559;
	  config.Alayer(2,2)=3.06448590934;
	  config.Alayer(2,3)=-2.64037530024;
	  config.Alayer(3,0)=0.833327936496;
	  config.Alayer(3,1)=-2.25743854559;
	  config.Alayer(3,2)=3.06448590934;
	  config.Alayer(3,3)=-2.64037530024;

	}
      else if (config.nlayer==5)
	{
	  config.Vlayer(0)=0.608;
	  config.Vlayer(1)=0.2184165;
	  config.Vlayer(2)=0.12102374;
	  config.Vlayer(3)=0.04255976;
	  config.Vlayer(4)=0.01;
	  config.alpha_layer(0)=1.0/1.00194135033;
	  config.alpha_layer(1)=6.2/1.18/1.07132849494;
	  config.alpha_layer(2)=68.0/0.91/2.3390977/1.0329289261;
	  config.alpha_layer(3)=6800.0/0.2184165/11.66905/4.02193787237;
	  config.alpha_layer(4)=1.0e20;

	  config.Alayer(0,0)=0.76764824;
	  config.Alayer(0,1)=-2.13955138;
	  config.Alayer(0,2)=2.56370476;
	  config.Alayer(0,3)=-2.19180146;
	  config.Alayer(1,0)=-0.53213871;
	  config.Alayer(1,1)=1.34851883;
	  config.Alayer(1,2)=-1.62609408;
	  config.Alayer(1,3)=-0.19028543;
	  config.Alayer(2,0)=0.25057814;
	  config.Alayer(2,1)=-0.66492431;
	  config.Alayer(2,2)=0.70707562;
	  config.Alayer(2,3)=-1.2927267;
	  config.Alayer(3,0)=-0.04054916;
	  config.Alayer(3,1)=0.11232971;
	  config.Alayer(3,2)=0.23075899;
	  config.Alayer(3,3)=-1.30253945;
	  config.Alayer(4,0)=-0.04054916;
	  config.Alayer(4,1)=0.11232971;
	  config.Alayer(4,2)=0.23075899;
	  config.Alayer(4,3)=-1.30253945;  	    
	}
      else if (config.nlayer>10)
	{
	  double sumVlayer=0.0;
	  for (ilayer=0;ilayer<config.nlayer-1;ilayer++)
	    {
	      config.Vlayer(ilayer)=6.0/pow(3.14148*(ilayer+1),2.0);
	      sumVlayer+=config.Vlayer(ilayer);
	      config.alpha_layer(ilayer)=pow(double(ilayer+1),2);
	      config.Alayer(ilayer,0)=-0.207952538556;
	      config.Alayer(ilayer,1)=0.861371078474;
	      config.Alayer(ilayer,2)=-0.109866578569;
	      config.Alayer(ilayer,3)=-1.54355196135;
	    }
	  ilayer=config.nlayer-1;
	  config.Vlayer(ilayer)=1.0-sumVlayer;
	  config.alpha_layer(ilayer)=pow(double(ilayer+1),2);
	  config.Alayer(ilayer,0)=-0.207952538556;
	  config.Alayer(ilayer,1)=0.861371078474;
	  config.Alayer(ilayer,2)=-0.109866578569;
	  config.Alayer(ilayer,3)=-1.54355196135;

	}

      else
	{
	  cout << "WARNING: parameters for a number of layers " 
	       << config.nlayer << " not defined." << endl;
	  exit(0);
	}
    }

  for (i=0;i<n;++i)
    {      
      surrogate[i].Ap_layer.resize(config.nbins,config.nlayer,config.max_number_of_phases);
      surrogate[i].Aaq_bins.resize(config.nbins);
      surrogate[i].Ap_layer_init.resize(config.nbins,config.nlayer,config.max_number_of_phases);
      surrogate[i].Ap_layer_init0.resize(config.nbins,config.nlayer,config.max_number_of_phases);
      for (b=0;b<config.nbins;++b)		  
        for (ilayer=0;ilayer<config.nlayer;++ilayer)
          for (iphase=0;iphase<config.max_number_of_phases;++iphase)
            surrogate[i].Ap_layer_init(b,ilayer,iphase)=0.0;
      surrogate[i].Aaq_bins_init.resize(config.nbins);
      surrogate[i].Aaq_bins_init0.resize(config.nbins);
      surrogate[i].gamma_org_layer.resize(config.nbins,config.nlayer,config.max_number_of_phases);
      surrogate[i].gamma_org_layer0.resize(config.nbins,config.nlayer,config.max_number_of_phases);
      surrogate[i].Xinit.resize(config.nbins,config.nlayer,config.max_number_of_phases);
      surrogate[i].gamma_aq_bins.resize(config.nbins);
      surrogate[i].LR.resize(config.nbins);
      surrogate[i].SRMR.resize(config.nbins);
      surrogate[i].tau_diffusion.resize(config.nbins,config.nlayer,config.max_number_of_phases);
      surrogate[i].tau_air.resize(config.nbins);
      surrogate[i].k1.resize(config.nbins,config.nlayer,config.max_number_of_phases,2);
      surrogate[i].Jdn.resize(config.nbins,config.nlayer,config.max_number_of_phases,2);
      surrogate[i].time.resize(config.nbins,config.nlayer,config.max_number_of_phases);
      surrogate[i].k1_aq.resize(config.nbins,2);
      surrogate[i].Jdn_aq.resize(config.nbins,2);
      surrogate[i].time_aq.resize(config.nbins);
      surrogate[i].Kp.resize(config.nbins,config.nlayer,config.max_number_of_phases);
      surrogate[i].Kaq.resize(config.nbins);
    }
  config.AQrho.resize(config.nbins);
}


void parameters(model_config& config, vector<species>& surrogate, vector<string> species_list_aer)
{    
  config.max_iter=1000;  //maximal number of iterations for the newton raphson method
  config.hygroscopicity=true; //Does hygroscopicity has to be computed?

  if (config.activity_model == "unifac")
    {
      config.tabulation_unifac = false;
      config.SR_ions = true;
      config.temperature_dependancy = true;
    }
  else
    {
      config.tabulation_unifac = true;
      config.SR_ions = false;
      config.temperature_dependancy = false;
    }

  config.tabulation_unifac = true;

  config.first_evaluation_activity_coefficients=false; //Use initial concentrations to compute activity coefficients
  config.LWClimit=0.01;      //LWC under which there is no aqueous phase
  config.rho_organic=1300.0; //volumic mass of the organic phase kg/m3
  config.compute_rho_aqueous=false;
  if (config.compute_rho_aqueous==false)
    config.rho_aqueous=1000.0; //volumic mass of the aqueous phase kg/m3

  config.compute_saturation=false;   //compute saturation
  config.compute_inorganic=false;
  if (config.compute_inorganic)
    config.compute_long_and_medium_range_interactions=true; //force to be used when inorganic are computed
  if (config.compute_long_and_medium_range_interactions==false) 
    config.SR_ions=false; // false if AIOMFAC is not used
  config.compute_organic=true;
  config.coupling_organic_inorganic=false;
  if (config.coupling_organic_inorganic==false)
    config.number_of_org_inorg_cycles=1;

  config.MOmin=1.0e-30;

  config.RHcoupling=0.98;

  if(config.compute_saturation and config.activity_model!="ideal")
    config.max_number_of_phases=2; //maximal number of organic phase
  else
    {
      config.compute_saturation=false; //no saturation is the system is ideal
      config.max_number_of_phases=1;
    }
  
  config.precision=0.0001;
  if (config.equilibrium==true)
    {
      config.precision=1e-4; //absolute precision under which the system has been solved
      config.initialized_saturation=false;
    }
  
  if (config.equilibrium==false)
    {
      config.relative_precision=0.001; //absolute precision under which the system has been solved
      config.first_evaluation_of_saturation=true; //Use initial concentrations to compute
      //phase separation?
      config.use_global_dynamic_parameters=false; //Assume the same composition over all bins
      // and layers
      config.compute_kelvin_effect=false;       //Should the kelvin effect be computed?
      config.tequilibrium=0.1;                 //time under which equilibrium is assumed
      config.EPSER=0.01;                       //relative difference of ros2
      config.deltatmin=1.0e-5; //1.0e-5;                 //minimal time step
                         //minimal concentrations of organics
      
      config.constant_dorg=true;

      config.explicit_representation=false;
      config.explicit_method=3;
      if (config.explicit_representation)
	config.nlayer=200;
	
      config.surface_tension_aq=72.0;  //surface tension of the organic phase
      config.surface_tension_org=24.0; //surface tension of the aqueous phase
      config.diameters.resize(config.nbins); //diameters of particles for each bin
      config.kp_low_volatility=0.001;
    }

  config.nh_inorg_init=1;
  config.nh_aq_init=1;
  config.nh_org_init=1;
  config.nh_max=10;
	  
  //create the vector of species and the various parameters of the model 
  creation_species(surrogate,species_list_aer); 
  system_coupling(config, surrogate);
  if (config.tabulation_unifac)
    {
      FILE* fp = NULL;
      fp = fopen( "unifac.csv", "rb" );
      if( fp != NULL )
        {
          fclose(fp);
          read_tab_unifac(config, surrogate); 
        }
      else
        {
          param_unifac(config, surrogate); 
        }
    }
  else
    param_unifac(config, surrogate); 
  system_aiomfac(config, surrogate);
  if (config.equilibrium==false)
    init_transfert_parameters(config, surrogate);
}


#endif
