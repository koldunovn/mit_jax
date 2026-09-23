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
C       G geometry  grid metrics, masks, Coriolis, angles, 3-D masks and mixing parameters (JAXDUMP_GEOM)
C       V vertical  1-D vertical grid packed into one 2-D record of tile 1 (JAXDUMP_GEOM)
C       R r* extra  rStarFacNm1, rStarExp, rStarDh*Dt, h0Fac, etaHnm1, hFac_surf, Bo_surf (JAXDUMP_GEOM)
C       X exchange  probe of every exch2 exchange type on index-coded fields (JAXDUMP_EXCH_PROBE)
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

      IF ( INDEX(groups,'G') .GT. 0 .OR. INDEX(groups,'V') .GT. 0
     &     .OR. INDEX(groups,'R') .GT. 0 )
     &  CALL JAXDUMP_GEOM( stage, groups, bi0, bj0, myIter, myThid )
      IF ( INDEX(groups,'X') .GT. 0 )
     &  CALL JAXDUMP_EXCH_PROBE( stage, myIter, myThid )
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

C---------------------------------------------------------------------------------------------------------------------
C     Dump a routine-local array shaped like a global field (all tiles), e.g. cg2d_b in SOLVE_FOR_PRESSURE.
      SUBROUTINE JAXDUMP_LOCAL( stage, fname, kind, arr, nz,
     &                          myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "JAXDUMP.h"
      CHARACTER*(*) stage, fname, kind
      INTEGER nz, myIter, myThid
      _RL arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nz,nSx,nSy)
      LOGICAL JAXDUMP_ACTIVE
      EXTERNAL JAXDUMP_ACTIVE
      IF ( .NOT. JAXDUMP_ACTIVE( myIter, myThid ) ) RETURN
      jd_seq = jd_seq + 1
      CALL JAXDUMP_FLD( stage, fname, kind, arr, nz, 0, 0,
     &                  myIter, myThid )
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Dump a 2-D slice of level k of a routine-local array inside a k loop; field name <fname>_k<kkk> (nz = 1).
      SUBROUTINE JAXDUMP_TILEK( stage, fname, kind, arr, k,
     &                          bi, bj, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "JAXDUMP.h"
      CHARACTER*(*) stage, fname, kind
      INTEGER k, bi, bj, myIter, myThid
      _RL arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      LOGICAL JAXDUMP_ACTIVE
      EXTERNAL JAXDUMP_ACTIVE
      INTEGER ILNBLNK
      EXTERNAL ILNBLNK
      CHARACTER*32 kname
      IF ( .NOT. JAXDUMP_ACTIVE( myIter, myThid ) ) RETURN
      jd_seq = jd_seq + 1
      WRITE(kname,'(2A,I3.3)') fname(1:ILNBLNK(fname)), '_k', k
      CALL JAXDUMP_WRITE( stage, kname, kind, arr, 1,
     &                    bi, bj, myIter, myThid )
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Geometry (G), packed vertical grid (V) and extra r* fields (R). Called from JAXDUMP_STATE (already active).
      SUBROUTINE JAXDUMP_GEOM( stage, groups, bi0, bj0, myIter,
     &                         myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "SURFACE.h"
#ifdef ALLOW_CTRL
# include "CTRL_OPTIONS.h"
# include "CTRL_FIELDS.h"
#endif
#ifdef ALLOW_MOM_COMMON
# include "MOM_COMMON_OPTIONS.h"
# include "MOM_VISC.h"
#endif
      CHARACTER*(*) stage, groups
      INTEGER bi0, bj0, myIter, myThid
      _RL vbuf(1-OLx:sNx+OLx,1-OLy:sNy+OLy,1,nSx,nSy)
      INTEGER i, j, bi, bj

      IF ( INDEX(groups,'G') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'xC','C',xC,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'yC','C',yC,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'xG','Z',xG,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'yG','Z',yG,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dxC','W',dxC,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dyC','S',dyC,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dxF','C',dxF,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dyF','C',dyF,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dxG','S',dxG,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dyG','W',dyG,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dxV','Z',dxV,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dyU','Z',dyU,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rA','C',rA,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rAw','W',rAw,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rAs','S',rAs,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rAz','Z',rAz,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dxC','W',recip_dxC,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dyC','S',recip_dyC,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dxF','C',recip_dxF,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dyF','C',recip_dyF,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dxG','S',recip_dxG,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dyG','W',recip_dyG,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dxV','Z',recip_dxV,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dyU','Z',recip_dyU,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_rA','C',recip_rA,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_rAw','W',recip_rAw,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_rAs','S',recip_rAs,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_rAz','Z',recip_rAz,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'R_low','C',R_low,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rLowW','W',rLowW,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rLowS','S',rLowS,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Ro_surf','C',Ro_surf,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'rSurfW','W',rSurfW,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'rSurfS','S',rSurfS,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'recip_Rcol','C',recip_Rcol,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'maskInC','C',maskInC,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'maskInW','W',maskInW,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'maskInS','S',maskInS,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'angleCosC','C',angleCosC,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'angleSinC','C',angleSinC,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'u2zonDir','W',u2zonDir,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'v2zonDir','S',v2zonDir,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'fCori','C',fCori,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'fCoriG','Z',fCoriG,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'fCoriCos','C',fCoriCos,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'tanPhiAtU','W',tanPhiAtU,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'tanPhiAtV','S',tanPhiAtV,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'maskC','C',maskC,Nr,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'maskW','W',maskW,Nr,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'maskS','S',maskS,Nr,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'recip_hFacW','W',recip_hFacW,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_hFacS','S',recip_hFacS,Nr,
     &                  bi0,bj0,myIter,myThid)
#ifdef ALLOW_3D_DIFFKR
       CALL JAXDUMP_FLD(stage,'diffKr','C',diffKr,Nr,bi0,bj0,myIter,
     &                  myThid)
#endif
#ifdef ALLOW_KAPGM_CONTROL
       CALL JAXDUMP_FLD(stage,'kapGM','C',kapGM,Nr,bi0,bj0,myIter,
     &                  myThid)
#endif
#ifdef ALLOW_KAPREDI_CONTROL
       CALL JAXDUMP_FLD(stage,'kapRedi','C',kapRedi,Nr,bi0,bj0,myIter,
     &                  myThid)
#endif
#ifdef ALLOW_3D_VISCA4
       CALL JAXDUMP_FLD(stage,'viscA4Dfld','C',viscA4Dfld,Nr,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'viscA4Zfld','Z',viscA4Zfld,Nr,bi0,bj0,
     &                  myIter,myThid)
#endif
#ifdef ALLOW_3D_VISCAH
       CALL JAXDUMP_FLD(stage,'viscAhDfld','C',viscAhDfld,Nr,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'viscAhZfld','Z',viscAhZfld,Nr,bi0,bj0,
     &                  myIter,myThid)
#endif
      ENDIF

      IF ( INDEX(groups,'V') .GT. 0 ) THEN
C      rows (j) of tile 1: 1 drF, 2 drC, 3 rC, 4 rF, 5 recip_drF, 6 recip_drC, 7 tRef, 8 sRef,
C      9 rVel2wUnit, 10 wUnit2rVel, 11 rhoFacC, 12 rhoFacF, 13 dBdrRef, 14 phiRef(1:Nr+1), 15 phiRef(Nr+2:2Nr+1)
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           vbuf(i,j,1,bi,bj) = 0. _d 0
          ENDDO
         ENDDO
        ENDDO
       ENDDO
       DO i = 1, Nr
        vbuf(i, 1,1,1,1) = drF(i)
        vbuf(i, 3,1,1,1) = rC(i)
        vbuf(i, 5,1,1,1) = recip_drF(i)
        vbuf(i, 7,1,1,1) = tRef(i)
        vbuf(i, 8,1,1,1) = sRef(i)
        vbuf(i,11,1,1,1) = rhoFacC(i)
        vbuf(i,13,1,1,1) = dBdrRef(i)
        vbuf(i,15,1,1,1) = phiRef(Nr+1+i)
       ENDDO
       DO i = 1, Nr+1
        vbuf(i, 2,1,1,1) = drC(i)
        vbuf(i, 4,1,1,1) = rF(i)
        vbuf(i, 6,1,1,1) = recip_drC(i)
        vbuf(i, 9,1,1,1) = rVel2wUnit(i)
        vbuf(i,10,1,1,1) = wUnit2rVel(i)
        vbuf(i,12,1,1,1) = rhoFacF(i)
        vbuf(i,14,1,1,1) = phiRef(i)
       ENDDO
       CALL JAXDUMP_FLD(stage,'vertical','V',vbuf,1,1,1,myIter,myThid)
      ENDIF

#ifdef NONLIN_FRSURF
      IF ( INDEX(groups,'R') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'rStarFacNm1C','C',rStarFacNm1C,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarFacNm1W','W',rStarFacNm1W,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarFacNm1S','S',rStarFacNm1S,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarExpC','C',rStarExpC,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarExpW','W',rStarExpW,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarExpS','S',rStarExpS,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarDhCDt','C',rStarDhCDt,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarDhWDt','W',rStarDhWDt,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarDhSDt','S',rStarDhSDt,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'pStarFacK','C',pStarFacK,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'h0FacC','C',h0FacC,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'h0FacW','W',h0FacW,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'h0FacS','S',h0FacS,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'etaHnm1','C',etaHnm1,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'hFac_surfC','C',hFac_surfC,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'hFac_surfW','W',hFac_surfW,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'hFac_surfS','S',hFac_surfS,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Bo_surf','C',Bo_surf,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_Bo','C',recip_Bo,1,
     &                  bi0,bj0,myIter,myThid)
      ENDIF
#endif
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Exchange probe: fill tile interiors with exact codes, halos with 0, run each exch2 exchange type used by the
C     model, dump the results with halos. code(u-like) = 1e6 + tile*1e4 + j*100 + i, code(v-like) = 2e6 + ...;
C     a negative halo value = sign flip, 0 = not filled. Arrays live in a local common (large).
      SUBROUTINE JAXDUMP_EXCH_PROBE( stage, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#ifdef ALLOW_EXCH2
# include "W2_EXCH2_SIZE.h"
# include "W2_EXCH2_TOPOLOGY.h"
#endif
      CHARACTER*(*) stage
      INTEGER myIter, myThid
      COMMON /JAXDUMP_PROBE/ pu, pv
      _RL pu(1-OLx:sNx+OLx,1-OLy:sNy+OLy,1,nSx,nSy)
      _RL pv(1-OLx:sNx+OLx,1-OLy:sNy+OLy,1,nSx,nSy)
      INTEGER m

      DO m = 1, 12
       CALL JAXDUMP_PROBE_FILL( pu, pv, myThid )
       IF ( m .EQ. 1 ) THEN
        CALL EXCH_XY_RL( pu, myThid )
        CALL JAXDUMP_FLD(stage,'xT','C',pu,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 2 ) THEN
        CALL EXCH_UV_XY_RL( pu, pv, .TRUE., myThid )
        CALL JAXDUMP_FLD(stage,'xUVs_u','W',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xUVs_v','S',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 3 ) THEN
        CALL EXCH_UV_XY_RL( pu, pv, .FALSE., myThid )
        CALL JAXDUMP_FLD(stage,'xUVn_u','W',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xUVn_v','S',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 4 ) THEN
        CALL EXCH_Z_3D_RL( pu, 1, myThid )
        CALL JAXDUMP_FLD(stage,'xZ','Z',pu,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 5 ) THEN
        CALL EXCH_UV_AGRID_3D_RL( pu, pv, .TRUE., 1, myThid )
        CALL JAXDUMP_FLD(stage,'xAs_u','C',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xAs_v','C',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 6 ) THEN
        CALL EXCH_UV_AGRID_3D_RL( pu, pv, .FALSE., 1, myThid )
        CALL JAXDUMP_FLD(stage,'xAn_u','C',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xAn_v','C',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 7 ) THEN
        CALL EXCH_UV_BGRID_3D_RL( pu, pv, .TRUE., 1, myThid )
        CALL JAXDUMP_FLD(stage,'xBs_u','Z',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xBs_v','Z',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 8 ) THEN
        CALL EXCH_UV_BGRID_3D_RL( pu, pv, .FALSE., 1, myThid )
        CALL JAXDUMP_FLD(stage,'xBn_u','Z',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xBn_v','Z',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 9 ) THEN
C       EXCH_S3D_RL works on (0:sNx+1,0:sNy+1) arrays (halo 1): not probed here
        CALL EXCH_3D_RL( pu, 1, myThid )
        CALL JAXDUMP_FLD(stage,'x3D','C',pu,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 10 ) THEN
        CALL EXCH_SM_3D_RL( pu, .TRUE., 1, myThid )
        CALL JAXDUMP_FLD(stage,'xSMs','C',pu,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 11 ) THEN
        CALL EXCH_UV_DGRID_3D_RL( pu, pv, .TRUE., 1, myThid )
        CALL JAXDUMP_FLD(stage,'xDs_u','S',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xDs_v','W',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 12 ) THEN
        CALL EXCH_UV_3D_RL( pu, pv, .TRUE., 1, myThid )
        CALL JAXDUMP_FLD(stage,'xUV3s_u','W',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xUV3s_v','S',pv,1,0,0,myIter,myThid)
       ENDIF
      ENDDO
      RETURN
      END

      SUBROUTINE JAXDUMP_PROBE_FILL( pu, pv, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#ifdef ALLOW_EXCH2
# include "W2_EXCH2_SIZE.h"
# include "W2_EXCH2_TOPOLOGY.h"
#endif
      INTEGER myThid
      _RL pu(1-OLx:sNx+OLx,1-OLy:sNy+OLy,1,nSx,nSy)
      _RL pv(1-OLx:sNx+OLx,1-OLy:sNy+OLy,1,nSx,nSy)
      INTEGER i, j, bi, bj, tile
      DO bj = 1, nSy
       DO bi = 1, nSx
#ifdef ALLOW_EXCH2
        tile = W2_myTileList(bi,bj)
#else
        tile = bi + (bj-1)*nSx
#endif
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          pu(i,j,1,bi,bj) = 0. _d 0
          pv(i,j,1,bi,bj) = 0. _d 0
          IF ( i.GE.1 .AND. i.LE.sNx .AND. j.GE.1 .AND. j.LE.sNy ) THEN
           pu(i,j,1,bi,bj) = 1000000 + tile*10000 + j*100 + i
           pv(i,j,1,bi,bj) = 2000000 + tile*10000 + j*100 + i
          ENDIF
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      RETURN
      END
