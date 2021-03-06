C-----------------------------------------------------------------------
C     Copyright (C) 2003-2007, ENPC - INRIA - EDF R&D
C     Author(s): Edouard Debry
C
C     This file is part of the Size Resolved Aerosol Model (SIREAM), a
C     component of the air quality modeling system Polyphemus.
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

      SUBROUTINE ETRCONC(neq,nbin_aer,q1,q,iq,couples_coag,
     s      first_index_coag,second_index_coag,
     s      coefficient_coag,QT,XSF,MSF,DSF,XSD,MSD,DSD)

C------------------------------------------------------------------------
C
C     -- DESCRIPTION
C
C     This subroutine solves a system of Ordinary Differential Equations
C     provided by the GDE for aerosols with the Explicit Trapezoidal Rule
C     algorithm (ETR).
C
C------------------------------------------------------------------------
C
C     -- INPUT VARIABLES
C
C     NEQ : number of equations.
C     nbin_aer: number of aerosol bins.
C     Q   : gas/aerosol concentrations ([\mu.g.m^-3]).
C     iq: index of aerosol species in q(*) vector.
C     couples_coag: coagulation couples for each bin.
C     first_index_coag: first index of coagulation couples.
C     second_index_coag: second index of coagulation couples.
C     coefficient_coag: coagulation partition coefficient ([adim]).
C     XSF: neperian logarithm of fixed aerosol bin mass ([adim]).
C     MSF: fixed aerosol bin dry mass ([\mu g]).
C     DSF: fixed aerosol bin dry diameter ([\mu m]).
C
C     -- INPUT/OUTPUT VARIABLES
C
C     Q   : gas/aerosol concentrations ([\mu.g.m^-3]).
C     XSD: neperian logarithm of moving aerosol bin mass ([adim]).
C     MSD: moving aerosol bin dry mass ([\mu g]).
C     DSD: moving aerosol bin dry diameter ([\mu m]).
C     QT: total aerosol concentration per bin ([\mu g.m^-3]).
C
C     -- OUTPUT VARIABLES
C
C     Q1 : first-order evaluation for concentrations ([\mu g.m^-3]).
C
C------------------------------------------------------------------------
C
C     -- REMARKS
C
C------------------------------------------------------------------------
C
C     -- MODIFICATIONS
C
C     2005/3/23: cleaning (Bruno Sportisse, CEREA).
C
C------------------------------------------------------------------------
C
C     -- AUTHOR(S)
C
C     2004: Edouard Debry, CEREA.
C
C
C------------------------------------------------------------------------

      IMPLICIT NONE

      INCLUDE 'param.inc'
      INCLUDE 'time.inc'

      INTEGER neq,nbin_aer
      DOUBLE PRECISION q(neq),q1(neq)
      INTEGER iq(NEXT,nbin_aer)

      INTEGER jj
      DOUBLE PRECISION dq1dt(neq),dq2dt(neq)
      DOUBLE PRECISION dtetr

      INTEGER couples_coag(nbin_aer)
      INTEGER first_index_coag(nbin_aer, 4 * nbin_aer)
      INTEGER second_index_coag(nbin_aer, 4 * nbin_aer)
      DOUBLE PRECISION coefficient_coag(nbin_aer, nbin_aer, nbin_aer)

      DOUBLE PRECISION XSF(nbin_aer),MSF(nbin_aer),DSF(nbin_aer)
      DOUBLE PRECISION XSD(nbin_aer),MSD(nbin_aer),DSD(nbin_aer)

      DOUBLE PRECISION QT(nbin_aer)

      DOUBLE PRECISION AA(NEXT,nbin_aer)

C     First step
      CALL FGDE(neq,nbin_aer,q,iq,dq1dt,couples_coag,
     s      first_index_coag,second_index_coag,
     s      coefficient_coag,QT,XSF,MSF,DSF,XSD,MSD,DSD,AA)

      DO jj=1,neq
         dq1dt(jj) = DMAX1( -0.95d0*q(jj)/DT2,
     &        DMIN1(dq1dt(jj), q(jj)/DT2) )
         q1(jj)=q(jj)+DT2*dq1dt(jj)
      END DO

C     Second step

      TIN2=TIN2+DT2

      CALL FGDE(neq,nbin_aer,q1,iq,dq2dt,couples_coag,
     s      first_index_coag,second_index_coag,
     s      coefficient_coag,QT,XSF,MSF,DSF,XSD,MSD,DSD,AA)

C     Update the concentrations

      dtetr=DT2*5.0D-01

      DO jj=1,neq
         dq2dt(jj) = DMAX1( -0.95d0*q(jj)/DT2,
     &        DMIN1(dq2dt(jj), q(jj)/DT2) )
         q(jj)=q(jj)+dtetr*(dq1dt(jj)+dq2dt(jj))
      END DO


      END
