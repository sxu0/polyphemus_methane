/*************************************************************************

Include file : bunifacparma.h

Purpose: Unifac parameters Type B (replace file read)

Include dependencies:  Included in Unidriver.c

Notes: 5 primary compounds (EPRI 99): C23H47COOH, C8H17CH=CHC7H14COOH,
       4-(2-propio)-syringone, C29H60, 2-carboxybenzoic acid
       + 5 Type B compounds at 25C
       (Data of Lyman, Reehl, Rosenblatt, 1990)

       NMOL and NFUNC need to match DIMMOL and DIMFUN (used in unidriver.c)

       Parameters inputted as in the original fortran input file, therefore
       a transpose is needed in the unidriver C program for matrices A and NU
       in order to pass them properly into the Fortran unifac routine.

       Orders of the functional groups:
         Groups              Subgroups
         C                   CH3
         C                   CH2
         C                   CH
         C                   C
         C=C                 H2C=CH
         C=C                 HC=CH
         C=C                 HC=C
         Aromatic Carbon     ACH
         Aromatic Carbon     AC
         Aromatic C-alkane   ACCH3
         Aromatic C-alkane   ACCH2
         OH                  OH
         Aromatic C-alcohol  ACOH
         Carbonyl            CH3CO
         Aldehyde            HCO
         Ether               -OCH3
         Carboxylic Acid     COOH
         Aromatic Nitro      ACNO2
         Nitrate             NO3

revision History:  1. Developed by Betty Pun, AER, December, 1999
	              under CARB funding
                   2. Modified by Betty Pun, AER, December, 1999
                      under CARB funding for Type B module
		   3. Changed surrogate compounds for MADRID 1.5
		      Betty Pun, July 2005


**************************************************************************/


#ifndef BUNIPARM_H
#define BUNIPARM_H

/* no. of molecules */
int NMOL = 12;

/* no. of functional groups */
int NFUNC = 18;

/* Z = 10 is a fixed parameter in Unifac */
double Z = 10.0;

/* original file input has temperature,
but temperature is in main input file now */

/* group volume parameters */
/* dimension of RG is the same as NFUNC */
double RG[DIMFUN] = {0.9011, 0.6744, 0.4469, 0.2196, 1.1167, 0.8886, 0.5313, 0.3562, 1.2663, 1.0396, 1.0000, 0.8952, 1.6724, 0.9980, 1.1450, 1.3013, 1.4199, 2.1246};

/* group surface area parameters */
/* dimension of QG is the same as NFUNC */
double QG[DIMFUN] = {0.8480, 0.5400, 0.2280, 0.0000, 0.8670, 0.6760, 0.4000, 0.1200, 0.9680, 0.6600, 1.2000, 0.6800, 1.4880, 0.9480, 1.0880, 1.2440, 1.1040, 1.8682};

/* no. of groups in each molecule*/
int NU[DIMMOL][DIMFUN] = {
{1, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
{1, 14, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
{0, 0 , 0, 0, 0, 0, 2, 2, 0, 1, 0, 1, 1, 0, 2, 0, 0, 0},
{2, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0 , 0, 0, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0},
{0, 0 , 0, 0, 0, 0, 3, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0},
{0, 0 , 0, 0, 0, 0, 3, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0},
{3, 5 , 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1},
{3, 4 , 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
{1, 1 , 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1},
{1, 2 , 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3},
{3, 2 , 3, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1}};

/* no. of groups in each molecule*/

double A[DIMFUN][DIMFUN] = {
  {0.0  ,  0.0   , 0.0   , 0.0   , 86.020, 86.020, 61.13,  61.13 , 76.50 ,76.50,  986.5,  1333.0 , 476.4  ,677   , 215.5 , 663.5 ,543, 500.95},
  {0.0  ,  0.0   , 0.0   , 0.0   , 86.020, 86.020, 61.13,  61.13 , 76.50 ,76.50,  986.5,  1333.0 , 476.4  ,677   , 215.5 , 663.5 ,543, 500.95},
  {0.0  ,  0.0   , 0.0   , 0.0   , 86.020, 86.020, 61.13,  61.13 , 76.50 ,76.50 , 986.5,  1333.0 , 476.4  ,677   , 215.5 , 663.5 ,543, 500.95},
  {0.0  ,  0.0   , 0.0   , 0.0   , 86.020, 86.020, 61.13,  61.13 , 76.50 ,76.50 , 986.5,  1333.0 , 476.4  ,677   , 215.5 , 663.5 ,543, 500.95},
  {-35.36 ,  -35.36  , -35.36  , -35.36  , 0.0    , 0.0    , 38.810,  38.810 , 74.15  ,74.15  , 524.1,  526.1 , 182.6  ,  448.8  , 214.5 , 318.9 ,0.0, 10326.0},
  {-35.36 ,  -35.36  , -35.36  , -35.36  , 0.0    , 0.0    , 38.810,  38.810 , 74.15  ,74.15  , 524.1,  526.1 , 182.6  , 448.8  , 214.5  ,318.9 ,0.0, 10326.0},
  {-11.12, -11.12, -11.12, -11.12, 3.4460, 3.4460, 0.0  ,  0.0   , 167.0   ,167.0   , 636.1,  1329.0  , 25.77 , 347.3   , 32.14 , 537.4, 194.9, 0.0},
  {-11.12, -11.12, -11.12, -11.12, 3.4460, 3.4460, 0.0  ,  0.0   , 167.0   ,167.0   , 636.1,  1329.0  , 25.77 , 347.3   , 32.14 , 537.4, 194.9, 0.0},
  {-69.70, -69.70, -69.70, -69.70, -113.6 , -113.6, -146.8, -146.8, 0.0   ,0.0   , 803.2,  884.9 , -52.10, 586.6   , 213.1 , 872.3, 4448, 0.0},
  {-69.70, -69.70, -69.70, -69.70, -113.6 , -113.6, -146.8, -146.8, 0.0   ,0.0   , 803.2,  884.9 , -52.10, 586.6   , 213.1 , 872.3, 4448, 0.0},
  {156.4 , 156.4 , 156.4 , 156.4 , 457.0  , 457.0 , 89.60 , 89.60,  25.82 ,25.82 , 0.0  ,  -259.7, 84.00 , -203.6 , 28.06 , 119.0, 157.1, 37.631},
  {275.8 , 275.8 , 275.8 , 275.8 , 217.5  , 217.5  , 25.34 , 25.34,  244.20 ,244.20 , -541.6, 0.0   , -356.1, -271.1   , -162.9   , 408.9  , -413.48, 0.0},
  {26.76 , 26.76 , 26.76 , 26.76 , 42.92 , 42.92 , 140.1 , 140.1,  365.8 ,365.8 , 164.5,  -133.1, 0.0   , -37.36, 5.202 , 669.4, 548.5, -197.93},
  {505.7 , 505.7 , 505.7 , 505.7 , 56.36 , 56.36 , 23.39 , 23.39,  106.0 , 106.0, 529.0,  -155.6, 128   , 0.0   , 304.1,  497.5,   0.0, 402.00},
  {83.36 , 83.36 , 83.36 , 83.36 , 26.51 , 26.51 , 52.13 , 52.13,  65.69 ,65.69 , 237.7,  -178.5, 191.1 , -7.838, 0.0   , 664.6, 155.11, 1131.1},
  {315.3 , 315.3 , 315.3 , 315.3 , 1264.0  , 1264.0  , 62.32 , 62.32,  89.80 ,89.80 , -151 ,  -11.00, -297.8, -165.5, -338.5, 0.0  , 0.0, -100.17},
  {5541  , 5541  , 5541  , 5541  , 0.0    , 0.0  , 1824.0  , 1824.0 ,  -127.8,-127.8, 561.6,  815.12, -101.5, 0.0   , 220.66, 0.0  , 0.0, 0.0},
  {-75.718, -75.718, -75.718, -75.718, -294.43, -294.43, 0.0, 0.0, 0.0, 0.0, 818.97, 0.0, 188.72, -179.38, -289.81, 1173.3, 0.0, 0.0}};

#endif

/********************END unifacparam.h**********************************/
