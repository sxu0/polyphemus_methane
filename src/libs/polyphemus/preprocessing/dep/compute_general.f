C-----------------------------------------------------------------------
C     Copyright (C) 2005-2007, ENPC - INRIA - EDF R&D
C
C     This file is part of the air quality modeling system Polyphemus.
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



      SUBROUTINE COMPUTE_DYNAMIC_VISCOSITY(Temperature,dyn_viscosity)
C------------------------------------------------------------------------
C
C     -- DESCRIPTION
C
C     This function computes the dynamic air viscosity with the
C     Sutherland law. Ref: Jacobson 1999, p92, eq. 4.55.
C
C------------------------------------------------------------------------
C
C     -- INPUT VARIABLES
C
C     Temperature : temperature ([K]).
C
C     -- OUTPUT VARIABLES
C
C     dyn_viscosity: air viscosity ([kg/m/s]).
C
C------------------------------------------------------------------------

      IMPLICIT NONE

      DOUBLE PRECISION Temperature, dyn_viscosity

      dyn_viscosity = 1.8325d-5 * ( 416.16d0 /
     $     ( Temperature + 120.d0) ) *
     $     ( Temperature / 296.16d0 )**1.5d0

      RETURN
      END


C------------------------------------------------------------------------
      subroutine compute_CC(LAMBDA,DP,CC)
C------------------------------------------------------------------------
C
C     -- DESCRIPTION
C
C     This function computes the correction Cunningham factor for a
C     particle of diameter D.
C     Pandis/Seinfeld 1998, page 464, (8:34).
C
C------------------------------------------------------------------------
C
C     -- INPUT VARIABLES
C
C     DP : particle diameter ([m]).
C     LAMBDA: Air free mean path ([m]).
C
C     -- OUTPUT VARIABLES
C
C     CC: Cunningham factor.
C
C------------------------------------------------------------------------

      IMPLICIT NONE

      DOUBLE PRECISION  LAMBDA,DP,CC

      CC=1.257D0 + 0.4D0 * DEXP(-1.1D0 * DP /(2.D0 * LAMBDA))
      CC=1.D0+(2.D0*LAMBDA/DP)*CC

      RETURN
      END

C------------------------------------------------------------------------
      subroutine compute_VSTOKES(DP,RHOP,CC,DLMUAIR,VSTOKES)
C------------------------------------------------------------------------
C
C     -- DESCRIPTION
C
C     This function computes the gravitational settling velocity for
C     a particle (density, diameter) with the Stokes formula.
C     Pandis/Seinfeld 1998, page 466, (8:42).
C
C------------------------------------------------------------------------
C
C     -- INPUT VARIABLES
C
C     DP      : particle diameter          ([m]).
C     RHOP    : particle density           ([kg/m3]).
C     CC      : Cunningham correction factor
C     DLMUAIR : air viscosity              ([kg/m/s]).
C
C     -- OUTPUT VARIABLES
C
C     VSTOKES : gravitational settling velocity ([m/s]).
C
C------------------------------------------------------------------------

      IMPLICIT NONE

      DOUBLE PRECISION DP,RHOP,CC,DLMUAIR,VSTOKES

C     Gravity acceleration. ([m/s^2])
      DOUBLE PRECISION g

      g = 9.81d0

      VSTOKES=DP*DP*RHOP*g*CC/(18.D0*DLMUAIR)

      RETURN
      END



C------------------------------------------------------------------------
      subroutine compute_CD(Re,CD)
C------------------------------------------------------------------------
C
C     -- DESCRIPTION
C
C     This function computes the drag coefficent as a function of the
C     Reynolds number.
C     Pandis/Seinfeld 1998, page 463, (8:32).
C
C------------------------------------------------------------------------
C
C     -- INPUT VARIABLES
C
C     Re : Reynolds Number.
C
C     -- OUTPUT VARIABLES
C
C     CD : drag coefficient.
C
C------------------------------------------------------------------------

      IMPLICIT NONE

      DOUBLE PRECISION Re,CD

      IF (Re.le.0.1D0) THEN
         CD=24.D0/Re
      ELSEIF ((Re.gt.0.1D0).and.(Re.le.2.D0)) THEN
         CD=24.D0/Re*(1.D0+0.1875D0*Re+0.05625D0*Re*Re*dlog(2.D0*Re))
      ELSEIF ((Re.gt.2.D0).and.(Re.le.500.D0)) THEN
         CD=24.D0/Re*(1.D0+0.15D0*Re**0.687D0)
      ELSEIF (Re.gt.500.D0) THEN
         CD=0.44D0
      ENDIF

      RETURN
      END


      SUBROUTINE COMPUTE_AIR_FREE_MEAN_PATH(Temperature,
     &     Pressure, air_free_mean_path, DLmuair)

C------------------------------------------------------------------------
C
C     -- DESCRIPTION
C
C     This function computes the free mean path for air molecules.
C     on the basis of thermodynamic variables. It also returns dynamic
C     viscosity.
C     Ref: Seinfeld & Pandis 1998, page 455 (8.6)
C
C------------------------------------------------------------------------
C
C     -- INPUT VARIABLES
C
C     Temperature : Temperature ([K]).
C     Pressure : Pressure    ([Pa]).
C
C     -- INPUT/OUTPUT VARIABLES
C
C
C     -- OUTPUT VARIABLES
C
C     AIR_FREE_MEAN_PATH : free mean path ([\micro m]).
C     DLMUAIR            : Dynamic viscosity ([kg/m/s]).
C
C------------------------------------------------------------------------

      IMPLICIT NONE

      DOUBLE PRECISION Temperature,Pressure
      DOUBLE PRECISION air_free_mean_path, DLMUAIR
C     Perfect gas constant. ([J.mol-1.K-1])
      DOUBLE PRECISION RGAS
C     Pi.
      DOUBLE PRECISION PI
C     Molar mass of air. ([kg.mol-1])
      DOUBLE PRECISION MMair

      RGAS = 8.314D0
      PI=3.14159265358979323846D0
      MMair = 2.897D-02

      call COMPUTE_DYNAMIC_VISCOSITY(Temperature,DLMUAIR)
      AIR_FREE_MEAN_PATH = DSQRT(PI*RGAS*Temperature/(2.d0*MMAIR))
     &     * DLMUAIR * 1.D6 / Pressure

      RETURN
      END
