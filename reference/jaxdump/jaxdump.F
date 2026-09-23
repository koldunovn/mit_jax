#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"
#ifdef ALLOW_EXF
# include "EXF_OPTIONS.h"
#endif
#ifdef ALLOW_GMREDI
# include "GMREDI_OPTIONS.h"
#endif
#ifdef ALLOW_GGL90
# include "GGL90_OPTIONS.h"
#endif
#ifdef ALLOW_SALT_PLUME
# include "SALT_PLUME_OPTIONS.h"
#endif

C=====================================================================================================================
C  jaxdump.F -- per-substep dump shim for the mitgcm-jax Fortran oracle (plan Task 5).
C
C  Enabled only when the environment variable JAXDUMP_DIR is set: with it unset every entry point returns at once,
C  and the model's output is byte-identical to the uninstrumented build (tested in scripts/tests/test_reference.py).
C  JAXDUMP_STEPS = myIter values separated by ',' or ':' (':' survives sbatch --export);
C  default nIter0, nIter0+1, nIter0+2. Every record of a step carries the step's START iteration.
C
C  Output: one stream file per (iteration, tile): JAXDUMP_DIR/jd_<myIter %010d>_t<tile %04d>.bin, records appended
C  in call order. Big-endian (the build uses -fconvert=big-endian). Record:
C     int32  magic = 1245990224 ('JDMP'), version = 1, myIter, seq
C     char*32 stage, char*32 field, char*4 kind (C W S Z: point type)
C     int32  nz, sNx, sNy, OLx, OLy, tile, face, tBasex, tBasey
C     real*8 values(1-OLx:sNx+OLx, 1-OLy:sNy+OLy, nz)   -- halos INCLUDED
C  Global key of value (i,j,k): face, iG = tBasex+i, jG = tBasey+j (halo points fall outside 1..tNx), k.
C  Reader: mitgcm_jax/io/dump.py.
C=====================================================================================================================

      BLOCK DATA JAXDUMP_BD
      IMPLICIT NONE
#include "JAXDUMP.h"
      DATA jd_inited /.FALSE./, jd_on /.FALSE./, jd_seq /0/
      END

C---------------------------------------------------------------------------------------------------------------------
      SUBROUTINE JAXDUMP_INIT( myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "JAXDUMP.h"
      INTEGER myThid
      EXTERNAL JAXDUMP_BD
      CHARACTER*(512) sbuf
      CHARACTER*(MAX_LEN_MBUF) msgBuf
      INTEGER ln, st, i, i0, n, ios
      INTEGER ILNBLNK
      EXTERNAL ILNBLNK

      jd_inited = .TRUE.
      jd_on = .FALSE.
      jd_nsteps = 0
      CALL GET_ENVIRONMENT_VARIABLE( 'JAXDUMP_DIR', jd_dir, ln, st )
      IF ( st .NE. 0 .OR. ln .EQ. 0 ) RETURN
      jd_on = .TRUE.
      CALL GET_ENVIRONMENT_VARIABLE( 'JAXDUMP_STEPS', sbuf, ln, st )
      IF ( st .NE. 0 .OR. ln .EQ. 0 ) THEN
        jd_nsteps = 3
        jd_steps(1) = nIter0
        jd_steps(2) = nIter0 + 1
        jd_steps(3) = nIter0 + 2
      ELSE
C       parse "a,b,c"
        i0 = 1
        DO i = 1, ln + 1
          IF ( i .GT. ln .OR. sbuf(i:i) .EQ. ','
     &                   .OR. sbuf(i:i) .EQ. ':' ) THEN
            IF ( i .GT. i0 ) THEN
              READ( sbuf(i0:i-1), *, IOSTAT=ios ) n
              IF ( ios .NE. 0 )
     &          STOP 'JAXDUMP: cannot parse JAXDUMP_STEPS'
              IF ( jd_nsteps .GE. jd_maxsteps )
     &          STOP 'JAXDUMP: too many JAXDUMP_STEPS'
              jd_nsteps = jd_nsteps + 1
              jd_steps(jd_nsteps) = n
            ENDIF
            i0 = i + 1
          ENDIF
        ENDDO
      ENDIF
      WRITE(msgBuf,'(2A,I4,A)') 'JAXDUMP: dumping to ',
     &   jd_dir(1:ILNBLNK(jd_dir)), jd_nsteps, ' iterations'
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, myThid )
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
      LOGICAL FUNCTION JAXDUMP_ACTIVE( myIter, myThid )
      IMPLICIT NONE
#include "JAXDUMP.h"
      INTEGER myIter, myThid
      INTEGER i
      IF ( .NOT. jd_inited ) CALL JAXDUMP_INIT( myThid )
      JAXDUMP_ACTIVE = .FALSE.
      IF ( .NOT. jd_on ) RETURN
      DO i = 1, jd_nsteps
        IF ( jd_steps(i) .EQ. myIter ) JAXDUMP_ACTIVE = .TRUE.
      ENDDO
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Write one tile's array (halos included) as one record.
      SUBROUTINE JAXDUMP_WRITE( stage, fname, kind, arr, nz,
     &                          bi, bj, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "JAXDUMP.h"
#ifdef ALLOW_EXCH2
# include "W2_EXCH2_SIZE.h"
# include "W2_EXCH2_TOPOLOGY.h"
#endif
      CHARACTER*(*) stage, fname, kind
      INTEGER nz, bi, bj, myIter, myThid
      _RL arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nz)
      INTEGER ILNBLNK
      EXTERNAL ILNBLNK
      CHARACTER*(600) fn
      CHARACTER*32 cstage, cname
      CHARACTER*4 ckind
      INTEGER*4 hdr(4), dims(9)
      REAL*8 buf(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nz)
      INTEGER iUnit, tile, face, tbx, tby, i, j, k

#ifdef ALLOW_EXCH2
      tile = W2_myTileList(bi,bj)
      face = exch2_myFace(tile)
      tbx  = exch2_tBasex(tile)
      tby  = exch2_tBasey(tile)
#else
      tile = bi + (bj-1)*nSx
      face = 0
      tbx  = 0
      tby  = 0
#endif
      WRITE(fn,'(2A,I10.10,A,I4.4,A)') jd_dir(1:ILNBLNK(jd_dir)),
     &     '/jd_', myIter, '_t', tile, '.bin'
      cstage = stage
      cname  = fname
      ckind  = kind
      hdr(1) = 1245990224
      hdr(2) = 1
      hdr(3) = myIter
      hdr(4) = jd_seq
      dims(1) = nz
      dims(2) = sNx
      dims(3) = sNy
      dims(4) = OLx
      dims(5) = OLy
      dims(6) = tile
      dims(7) = face
      dims(8) = tbx
      dims(9) = tby
      DO k = 1, nz
       DO j = 1-OLy, sNy+OLy
        DO i = 1-OLx, sNx+OLx
         buf(i,j,k) = arr(i,j,k)
        ENDDO
       ENDDO
      ENDDO
      CALL MDSFINDUNIT( iUnit, myThid )
      OPEN( iUnit, FILE=fn(1:ILNBLNK(fn)), ACCESS='STREAM',
     &      FORM='UNFORMATTED', STATUS='UNKNOWN', POSITION='APPEND' )
      WRITE( iUnit ) hdr, cstage, cname, ckind, dims, buf
      CLOSE( iUnit )
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Dump a global field (all tiles, or only tile bi0,bj0 when bi0 > 0).
      SUBROUTINE JAXDUMP_FLD( stage, fname, kind, fld, nz,
     &                        bi0, bj0, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "JAXDUMP.h"
      CHARACTER*(*) stage, fname, kind
      INTEGER nz, bi0, bj0, myIter, myThid
      _RL fld(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nz,nSx,nSy)
      INTEGER bi, bj
      DO bj = 1, nSy
       DO bi = 1, nSx
        IF ( bi0 .LE. 0 .OR. (bi .EQ. bi0 .AND. bj .EQ. bj0) )
     &    CALL JAXDUMP_WRITE( stage, fname, kind, fld(1-OLx,1-OLy,1,bi,
     &   bj),
     &                        nz, bi, bj, myIter, myThid )
       ENDDO
      ENDDO
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Dump the field groups named by the letters of `groups` at `stage` (all tiles if bi0 <= 0, else tile bi0,bj0):
C       d dynamics  u,v,w, etaN, etaH, dEtaHdt          t tracers   theta, salt
C       a AB histories and tendencies gU gV guNm gvNm gtNm gsNm (AB3: both levels)
C       r r*/hFac   hFacC/W/S, rStarFacC/W/S, recip_hFacC
C       f forcing   surfaceForcingU/V/T/S, fu, fv, Qnet, Qsw, EmPmR, saltFlux, pLoad, phi0surf, sIceLoad
C       m mixing    rhoInSitu, IVDConvCount, hMixLayer, totPhiHyd, phiHydLow, diffKr
C       k GGL90     GGL90TKE, GGL90viscArU/V, GGL90diffKr
C       g GM/Redi   Kwx Kwy Kwz Kux Kvy Kuz Kvz GM_PsiX GM_PsiY
C       p salt plume saltPlumeDepth, saltPlumeFlux
C       c cg2d      aW2d, aS2d, aC2d (operator), pW, pS, pC (preconditioner)
C       x EXF       ustress, vstress, hflux, swflux, sflux, saltflx, apressure (+ spflx with READIN_SALT_PLUME_FLUX)
      SUBROUTINE JAXDUMP_STATE( stage, groups, bi0, bj0, myIter,
     &   myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "SURFACE.h"
#include "FFIELDS.h"
#include "CG2D.h"
#ifdef ALLOW_GGL90
# include "GGL90.h"
#endif
#ifdef ALLOW_GMREDI
# include "GMREDI.h"
#endif
#ifdef ALLOW_SALT_PLUME
# include "SALT_PLUME.h"
#endif
#ifdef ALLOW_EXF
# include "EXF_PARAM.h"
# include "EXF_FIELDS.h"
#endif
#include "JAXDUMP.h"
      CHARACTER*(*) stage, groups
      INTEGER bi0, bj0, myIter, myThid
      LOGICAL JAXDUMP_ACTIVE
      EXTERNAL JAXDUMP_ACTIVE
      INTEGER m
      CHARACTER*8 suf

      IF ( .NOT. JAXDUMP_ACTIVE( myIter, myThid ) ) RETURN
      jd_seq = jd_seq + 1

      IF ( INDEX(groups,'d') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'uVel','W',uVel,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'vVel','S',vVel,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'wVel','C',wVel,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'etaN','C',etaN,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'etaH','C',etaH,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dEtaHdt','C',dEtaHdt,1,bi0,bj0,myIter,
     &                  myThid)
      ENDIF
      IF ( INDEX(groups,'t') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'theta','C',theta,Nr,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'salt','C',salt,Nr,bi0,bj0,myIter,myThid)
      ENDIF
      IF ( INDEX(groups,'a') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'gU','W',gU,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'gV','S',gV,Nr,bi0,bj0,myIter,myThid)
#ifdef ALLOW_ADAMSBASHFORTH_3
       DO m = 1, 2
        WRITE(suf,'(A,I1)') '_', m
        CALL JAXDUMP_FLD(stage,'guNm'//suf(1:2),'W',guNm(1-OLx,1-OLy,1,
     &   1,1,m),
     &                   Nr,bi0,bj0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'gvNm'//suf(1:2),'S',gvNm(1-OLx,1-OLy,1,
     &   1,1,m),
     &                   Nr,bi0,bj0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'gtNm'//suf(1:2),'C',gtNm(1-OLx,1-OLy,1,
     &   1,1,m),
     &                   Nr,bi0,bj0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'gsNm'//suf(1:2),'C',gsNm(1-OLx,1-OLy,1,
     &   1,1,m),
     &                   Nr,bi0,bj0,myIter,myThid)
       ENDDO
#else
       CALL JAXDUMP_FLD(stage,'guNm1','W',guNm1,Nr,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'gvNm1','S',gvNm1,Nr,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'gtNm1','C',gtNm1,Nr,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'gsNm1','C',gsNm1,Nr,bi0,bj0,myIter,
     &   myThid)
#endif
      ENDIF
      IF ( INDEX(groups,'r') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'hFacC','C',hFacC,Nr,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'hFacW','W',hFacW,Nr,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'hFacS','S',hFacS,Nr,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'recip_hFacC','C',recip_hFacC,Nr,
     &                  bi0,bj0,myIter,myThid)
#ifdef NONLIN_FRSURF
       CALL JAXDUMP_FLD(stage,'rStarFacC','C',rStarFacC,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarFacW','W',rStarFacW,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarFacS','S',rStarFacS,1,bi0,bj0,
     &                  myIter,myThid)
#endif
      ENDIF
      IF ( INDEX(groups,'f') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'surfaceForcingU','W',surfaceForcingU,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'surfaceForcingV','S',surfaceForcingV,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'surfaceForcingT','C',surfaceForcingT,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'surfaceForcingS','C',surfaceForcingS,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'fu','W',fu,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'fv','S',fv,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Qnet','C',Qnet,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Qsw','C',Qsw,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'EmPmR','C',EmPmR,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'saltFlux','C',saltFlux,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'pLoad','C',pLoad,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'phi0surf','C',phi0surf,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'sIceLoad','C',sIceLoad,1,bi0,bj0,
     &                  myIter,myThid)
      ENDIF
      IF ( INDEX(groups,'m') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'rhoInSitu','C',rhoInSitu,Nr,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'IVDConvCount','C',IVDConvCount,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'hMixLayer','C',hMixLayer,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'totPhiHyd','C',totPhiHyd,Nr,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'phiHydLow','C',phiHydLow,1,bi0,bj0,
     &                  myIter,myThid)
#ifdef ALLOW_3D_DIFFKR
       CALL JAXDUMP_FLD(stage,'diffKr','C',diffKr,Nr,bi0,bj0,
     &                  myIter,myThid)
#endif
      ENDIF
      IF ( INDEX(groups,'c') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'aW2d','W',aW2d,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'aS2d','S',aS2d,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'aC2d','C',aC2d,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'pW','W',pW,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'pS','S',pS,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'pC','C',pC,1,bi0,bj0,myIter,myThid)
      ENDIF
#ifdef ALLOW_GGL90
      IF ( INDEX(groups,'k') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'GGL90TKE','C',GGL90TKE,Nr,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'GGL90viscArU','W',GGL90viscArU,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'GGL90viscArV','S',GGL90viscArV,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'GGL90diffKr','C',GGL90diffKr,Nr,
     &                  bi0,bj0,myIter,myThid)
      ENDIF
#endif
#ifdef ALLOW_GMREDI
      IF ( INDEX(groups,'g') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'Kwx','C',Kwx,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Kwy','C',Kwy,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Kwz','C',Kwz,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Kux','W',Kux,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Kvy','S',Kvy,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Kuz','W',Kuz,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Kvz','S',Kvz,Nr,bi0,bj0,myIter,myThid)
#ifdef GM_BOLUS_ADVEC
       CALL JAXDUMP_FLD(stage,'GM_PsiX','W',GM_PsiX,Nr,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'GM_PsiY','S',GM_PsiY,Nr,bi0,bj0,
     &                  myIter,myThid)
#endif
      ENDIF
#endif
#ifdef ALLOW_SALT_PLUME
      IF ( INDEX(groups,'p') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'saltPlumeDepth','C',SaltPlumeDepth,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'saltPlumeFlux','C',saltPlumeFlux,1,
     &                  bi0,bj0,myIter,myThid)
      ENDIF
#endif
#ifdef ALLOW_EXF
      IF ( INDEX(groups,'x') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'ustress','W',ustress,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'vstress','S',vstress,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'hflux','C',hflux,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'swflux','C',swflux,1,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'sflux','C',sflux,1,bi0,bj0,myIter,myThid)
# ifdef ALLOW_SALTFLX
       CALL JAXDUMP_FLD(stage,'saltflx','C',saltflx,1,bi0,bj0,
     &                  myIter,myThid)
# endif
# ifdef ATMOSPHERIC_LOADING
       CALL JAXDUMP_FLD(stage,'apressure','C',apressure,1,bi0,bj0,
     &                  myIter,myThid)
# endif
# ifdef READIN_SALT_PLUME_FLUX
       CALL JAXDUMP_FLD(stage,'spflx','C',spflx,1,bi0,bj0,myIter,myThid)
# endif
      ENDIF
#endif
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Dump a routine-local array of ONE tile (e.g. sigmaX(i,j,k) inside a bi,bj loop).
      SUBROUTINE JAXDUMP_TILE( stage, fname, kind, arr, nz,
     &                         bi, bj, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "JAXDUMP.h"
      CHARACTER*(*) stage, fname, kind
      INTEGER nz, bi, bj, myIter, myThid
      _RL arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nz)
      LOGICAL JAXDUMP_ACTIVE
      EXTERNAL JAXDUMP_ACTIVE
      IF ( .NOT. JAXDUMP_ACTIVE( myIter, myThid ) ) RETURN
      jd_seq = jd_seq + 1
      CALL JAXDUMP_WRITE( stage, fname, kind, arr, nz,
     &                    bi, bj, myIter, myThid )
      RETURN
      END
