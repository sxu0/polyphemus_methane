C-----------------------------------------------------------------------
C     Copyright (C) 2003-2007, ENPC - INRIA - EDF R&D
C     Author(s): Edouard Debry
C
C     This file is part of the Size Resolved Aerosol Model (SIREAM),
C     which is a component of the air quality modeling system
C     Polyphemus.
C
C     Polyphemus is developed in the INRIA - ENPC joint project-team
C     CLIME and in the ENPC - EDF R&D joint laboratory CEREA.
C
C     Polyphemus is free software; you can redistribute it and/or modify
C     it under the terms of the GNU General Public License as published
C     by the Free Software Foundation; either version 2 of the License,
C     or (at your option) any later version.
C
C     Polyphemus is distributed in the hope that it will be useful, but
C     WITHOUT ANY WARRANTY; without even the implied warranty of
C     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
C     General Public License for more details.
C
C     For more information, visit the Polyphemus web site:
C     http://cerea.enpc.fr/polyphemus/
C-----------------------------------------------------------------------

C*************************************************
C
C     LIST OF POINTERS
C
C*************************************************
      INTEGER IH,INa,INH4,ICl,ISO4,IHSO4,INO3,
     &     IH2O,INH3,IHCl,IHNO3,IOH
      INTEGER SNaNO3,SNH4NO3,SNACl,SNH4Cl,SLC,
     &     SNa2SO4,SNH42S4,SNaHSO4,SNH4HS4
C*****
      PARAMETER (IH   =1,
     &     INa  =2,
     &     INH4 =3,
     &     ICl  =4,
     &     ISO4 =5,
     &     IHSO4=6,
     &     INO3 =7,
     &     IH2O =8,
     &     INH3 =9,
     &     IHCl =10,
     &     IHNO3=11,
     &     IOH  =12)

      PARAMETER (SNaNO3 =13,
     &     SNH4NO3=14,
     &     SNACl  =15,
     &     SNH4Cl =16,
     &     SNa2SO4=17,
     &     SNH42S4=18,
     &     SNaHSO4=19,
     &     SNH4HS4=20,
     &     SLC    =21)

C*************************************************