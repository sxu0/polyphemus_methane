SUBROUTINE EULER_NUMBER_NORM(ns, nesp, eh2o, dbound, grand, alpha, &
     fixed_diameter, diameter_before_redist, X, log_fixed_diameter, &
     fixed_density_aer, idens, kloc, LMD, DQLIMIT, rho, Qesp, N)

!!$------------------------------------------------------------------------
!!$
!!$     -- INPUT VARIABLES
!!$
!!$
!!$     ns             : number of sections
!!$     nesp           : number of species
!!$     dbound         : list of limit bound diameter [\mu m]
!!$     grand          : list of 0 or 1
!!$                      1 = cutting with the upper box
!!$                      0 = cutting with the lower box
!!$     alpha          : list of fraction of each species in Q
!!$     diameter_before_redist           : list of mean diameter after condensation/evaporation
!!$     kloc           : list of bin where is diameter_before_redist
!!$     log_fixed_diameter        : log(fixed_diameter)
!!$     X              : log(diameter_before_redist)
!!$     fixed_diameter              : list of mean diameter before condensation/evaporation
!!$     j              : time integration
!!$     section_pass   : bin include 100nm
!!$     LMD            : list of liquid mass density of each species
!!$
!!$     -- VARIABLES
!!$
!!$     Q              : Mass concentration
!!$     N_esp          : Number concentration by bin and species
!!$     rho            : density per bin
!!$     Eps_machine    : tolerance due to the lack of precision of the machine
!!$     Ndonne_esp     : Temporary number concentration
!!$     frac           : fraction define by X and log_fixed_diameter
!!$     Nd             : fraction of number concentration give at the adjacent bin
!!$
!!$     -- INPUT/OUTPUT VARIABLES
!!$
!!$     N            : Number concentration by bin
!!$     Qesp         : Mass concentration by bin and species
!!$
!!$     -- OUTPUT VARIABLES
!!$
!!$
!!$------------------------------------------------------------------------

  IMPLICIT NONE
  INCLUDE '../INC/parameuler.inc'

  ! ------ Input
  INTEGER, INTENT(in) :: ns, nesp
  INTEGER, DIMENSION(ns), INTENT(in) :: grand
  DOUBLE PRECISION, DIMENSION(ns), INTENT(in) :: X, log_fixed_diameter
  DOUBLE PRECISION, DIMENSION(ns), INTENT(in) :: fixed_diameter , diameter_before_redist
  DOUBLE PRECISION, DIMENSION(ns+1), INTENT(in) ::dbound
  INTEGER, DIMENSION(ns), INTENT(in) :: kloc
  DOUBLE PRECISION, DIMENSION(ns, nesp), INTENT(in) :: alpha
  DOUBLE PRECISION, DIMENSION(nesp), INTENT(in) :: LMD
  integer eh2o
  DOUBLE PRECISION, INTENT(in) :: fixed_density_aer
  INTEGER, INTENT(in) :: idens

  ! ------ Input/Output
  DOUBLE PRECISION, DIMENSION(ns), INTENT(inout) :: N
  DOUBLE PRECISION, DIMENSION(ns, nesp), INTENT(inout) :: Qesp

  ! ------
  INTEGER k, jesp
  DOUBLE PRECISION, DIMENSION(ns) :: rho
  DOUBLE PRECISION, DIMENSION(ns) :: Q, QT
  DOUBLE PRECISION, DIMENSION(ns, nesp) :: N_esp, Ndonne_esp
  DOUBLE PRECISION Nd, frac
  DOUBLE PRECISION, DIMENSION(nesp) :: QTOTin, QTOTout

  Q = 0.d0
  DO k = 1,ns
     DO jesp = 1, nesp
        if (jesp .ne. eh2o) then
        Q(k) = Q(k) + Qesp(k, jesp)
        endif
        N_esp(k, jesp) = alpha(k,jesp) * N(k)
     ENDDO
  ENDDO

  !***** Calcul total mass per species

  QTOTin=0.d0
  DO jesp = 1, nesp
     DO k = 1, ns
        QTOTin(jesp) = QTOTin(jesp) + Qesp(k, jesp)
     ENDDO
  ENDDO

  DO k = 1,ns
     DO jesp = 1, nesp
        Ndonne_esp(k, jesp) = 0d0
     ENDDO
  ENDDO

  DO k = 1,ns

     IF (grand(k) == 0)THEN

        IF (kloc(k) .NE. 1) THEN
           frac = (log_fixed_diameter(k) - X(k))/ &
                (log_fixed_diameter(k) - DLOG10(fixed_diameter(kloc(k)-1)))
        ELSE
           frac = (log_fixed_diameter(k) - X(k))/ &
                (log_fixed_diameter(k) - DLOG10(dbound(1)))/2.d0
        ENDIF
           IF (frac .GT. 1) THEN
           PRINT * , "In SIREAM/euler_number_norm.f90: frac > 1."
              STOP
           ENDIF

        DO jesp = 1, nesp

           Nd = N_esp(k, jesp) * frac

           IF (kloc(k) .NE. 1) THEN
              Ndonne_esp(kloc(k)-1, jesp)  = &
                   Ndonne_esp(kloc(k)-1, jesp) + Nd
              Ndonne_esp(kloc(k), jesp) = &
                   Ndonne_esp(kloc(k), jesp) + N_esp(k, jesp) - Nd
           ELSE
              Ndonne_esp(kloc(k), jesp) = &
                   Ndonne_esp(kloc(k), jesp) + N_esp(k, jesp)
           ENDIF
           N_esp(k, jesp) = 0.d0
        ENDDO

     ELSE
        IF (kloc(k) .NE. ns) THEN
           frac = (X(k) - log_fixed_diameter(k))/ &
                (DLOG10(fixed_diameter(kloc(k)+1)) - log_fixed_diameter(k))
        ELSE
           frac = (X(k) - log_fixed_diameter(k))/ &
                (DLOG10(dbound(ns+1)) - log_fixed_diameter(k))/2.d0
        ENDIF
           IF (frac .GT. 1) THEN
           PRINT * , "In SIREAM/euler_number_norm.f90: frac > 1."
              STOP
           ENDIF

        DO jesp = 1, nesp

           Nd =  N_esp(k, jesp) * frac

           IF (kloc(k) .NE. ns) THEN
              Ndonne_esp(kloc(k)+1,jesp) = &
                   Ndonne_esp(kloc(k)+1, jesp) + Nd
              Ndonne_esp(kloc(k), jesp) = &
                   Ndonne_esp(kloc(k), jesp) + N_esp(k, jesp) - Nd
           ELSE
              Ndonne_esp(kloc(k), jesp) = &
                   Ndonne_esp(kloc(k),jesp) + N_esp(k, jesp)
           ENDIF
           N_esp(k, jesp)=0.d0
        ENDDO
     ENDIF

  ENDDO



  N = 0.d0
  Q = 0.d0
  DO k = 1,ns
     DO jesp = 1, nesp
        N_esp(k, jesp) = N_esp(k, jesp) + Ndonne_esp(k, jesp)
        if (jesp .ne. eh2o) then
        N(k) = N(k) + N_esp(k, jesp)
        endif
     ENDDO

     !***** Recalculation of mass concentration from number concentration
     IF (IDENS .EQ. 1) THEN
        CALL COMPUTE_DENSITY(ns,nesp, eh2o, TINYN,N_esp,LMD,k,rho(k))
     ELSE
        rho(k) = fixed_density_aer
     ENDIF
     DO jesp = 1, nesp
        Qesp(k, jesp) = rho(k) * (PI/6D0) * N_esp(k,jesp) &
             * (fixed_diameter(k)*fixed_diameter(k)*fixed_diameter(k))
        if (jesp .ne. eh2o) then
        Q(k) = Q(k) + Qesp(k,jesp)
        endif
     ENDDO
  ENDDO



  QTOTout = 0.d0
  DO jesp = 1, nesp
     do k = 1, ns
        QTOTout(jesp) = QTOTout(jesp) + Qesp(k, jesp)
     enddo
  enddo

!*** Normalization
  do k = 1, ns
     DO jesp=1, nesp
        if (abs(QTOTin(jesp)-QTOTout(jesp)) .GT. DQLIMIT &
            .and. QTOTout(jesp) .GT. 0d0 ) then
           Qesp(k, jesp)  =  Qesp(k, jesp)*QTOTin(jesp)/QTOTout(jesp)
        endif
     enddo
  ENDDO

  N = 0.D0
  Q = 0.D0
  do k = 1, ns
     do jesp = 1, nesp-1
        Q(k) = Q(k) +  Qesp(k,jesp)
     enddo
     IF (IDENS .EQ. 1) THEN
        CALL COMPUTE_DENSITY(ns, nesp, eh2o, TINYM, Qesp, LMD, k, rho(k))
     ELSE
        rho(k) = fixed_density_aer
     ENDIF
     N(k) = Q(k) * 6.D0 / (PI * rho(k) * fixed_diameter(k) * fixed_diameter(k) * fixed_diameter(k))

  enddo


  CALL TEST_MASS_NB(ns,nesp,Q,N,Qesp)

END SUBROUTINE EULER_NUMBER_NORM
