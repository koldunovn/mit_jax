C     JAXDUMP.h -- state of the mitgcm-jax per-substep dump shim (reference/jaxdump/jaxdump.F).
C     jd_on      :: dumps enabled (JAXDUMP_DIR set and non-empty)
C     jd_inited  :: environment has been read
C     jd_nsteps  :: number of iterations to dump; jd_steps :: their myIter values (JAXDUMP_STEPS)
C     jd_seq     :: running record counter of this process (same for all tiles of one call)
C     jd_dir     :: output directory
      INTEGER jd_maxsteps
      PARAMETER ( jd_maxsteps = 200 )
      COMMON /JAXDUMP_L/ jd_on, jd_inited
      LOGICAL jd_on, jd_inited
      COMMON /JAXDUMP_I/ jd_nsteps, jd_steps, jd_seq
      INTEGER jd_nsteps, jd_steps(jd_maxsteps), jd_seq
      COMMON /JAXDUMP_C/ jd_dir
      CHARACTER*(512) jd_dir
