-- =============================================================================
-- OPEN OPPS extraction — every currently open opportunity, for scoring.
-- Save output as: data/input/open_opps.csv
--
-- Feature CTEs below (prior account wins, activity, contacts) are copied
-- verbatim from closed_opps.sql so features mean the same thing at train
-- and score time. If you change one, change both.
--
--   - CYCLE_LENGTH_DAYS = time to REACH the current stage (same definition
--     as training: start of the chronologically last stage row).
--   - DAYS_OPEN = days since TRUE_CYCLE_START; becomes DAYS_IN_SUSPECT
--     for opps that have never left Suspect.
--   - OWNER_IS_ACTIVE / OWNER_IS_NON_HUMAN drive the scoring gate
--     (inactive human owner -> skip scoring, send reassignment alert).
--     These are live-scoring only — never model features.
--   - IS_PROSPECT_DESK / IS_DEAD_QUEUE are output flags only.
--     VERIFY the Prospect Desk owner name matches the ILIKE pattern below.
--   - Requires OPTYS_GOLD.OWNERID; adjust the user_v join if named differently.
-- =============================================================================

-- Needed by won_history (prior account wins): closing date of every closed deal.
WITH true_close AS (
    SELECT
        ID AS OPPORTUNITY_ID,
        SLICE_START AS TRUE_CLOSE_DATE,
        STAGENAME AS TRUE_CLOSE_STAGENAME
    FROM DATALAB_SANDBOX.GS_REPORTING.OPTYS_STAGE_DURATION_WEIGHTED
    WHERE TRIM(STAGENAME) IN ('Closed Won / Implemented', 'Closed Lost.')
      AND IS_CURRENT = TRUE
),

opp_true_start AS (
    SELECT DISTINCT
        ID AS OPPORTUNITY_ID,
        TRUE_CYCLE_START
    FROM DATALAB_SANDBOX.GS_REPORTING.OPTYS_STAGE_DURATION_WEIGHTED
),

-- Chronologically last real-funnel stage row = the current stage for an open opp.
current_stage AS (
    SELECT
        ID AS OPPORTUNITY_ID,
        SLICE_START AS CURRENT_STAGE_START
    FROM DATALAB_SANDBOX.GS_REPORTING.OPTYS_STAGE_DURATION_WEIGHTED
    WHERE STAGE_NUMBER <= 5
      AND SLICE_START >= TRUE_CYCLE_START
    QUALIFY ROW_NUMBER() OVER (PARTITION BY ID ORDER BY SLICE_START DESC) = 1
),

opp_base AS (
    SELECT
        g.ID AS OPPORTUNITY_ID,
        g.ACCOUNTID,
        g.VERTICAL,
        g.SECTOR,
        g.AMOUNT,
        g.HEADCOUNT,
        g.INITIATIVE__C,
        g.STAGENAME AS CURRENT_STAGENAME,
        g.OWNER_NAME,
        rd.RECORDTYPE_NAME,
        UPPER(TRIM(a.BILLINGSTATE)) AS BILLINGSTATE,
        CASE WHEN g.OWNER_NAME = 'B2B_REMOTE GEO' THEN 1 ELSE 0 END AS IS_REMOTE_GEO,
        CASE WHEN g.OWNER_NAME ILIKE '%PROSPECT DESK%' THEN 1 ELSE 0 END AS IS_PROSPECT_DESK,
        CASE WHEN g.OWNER_NAME = 'B2B DEAD QUEUE' THEN 1 ELSE 0 END AS IS_DEAD_QUEUE,
        IFF(u.ISACTIVE, 1, 0) AS OWNER_IS_ACTIVE,
        IFF(u.IS_NON_HUMAN_USER__C, 1, 0) AS OWNER_IS_NON_HUMAN,
        ts.TRUE_CYCLE_START,
        GREATEST(DATEDIFF('day', ts.TRUE_CYCLE_START, cs.CURRENT_STAGE_START), 0) AS CYCLE_LENGTH_DAYS,
        DATEDIFF('day', ts.TRUE_CYCLE_START, CURRENT_DATE()) AS DAYS_OPEN,
        CASE
            WHEN COALESCE(g.AMOUNT, 0) = 0 THEN '$0'
            WHEN g.AMOUNT <= 5000 THEN '1-5K'
            WHEN g.AMOUNT <= 100000 THEN '5K-100K'
            WHEN g.AMOUNT <= 750000 THEN '100K-750K'
            ELSE '750K+'
        END AS AMOUNT_BUCKET
    FROM DATALAB_SANDBOX.GS_REPORTING.OPTYS_GOLD g
    JOIN opp_true_start ts      ON ts.OPPORTUNITY_ID = g.ID
    LEFT JOIN current_stage cs  ON cs.OPPORTUNITY_ID = g.ID
    LEFT JOIN DATALAB_SANDBOX.GS_SILVER.OPTYS_DIM_RECORD_DETAILS rd ON rd.ID = g.ID
    LEFT JOIN EDH.SFDC.ACCOUNT_V a ON a.ID = g.ACCOUNTID
    LEFT JOIN EDH.SFDC.USER_V u    ON u.ID = g.OWNERID
    WHERE TRIM(g.STAGENAME) NOT IN ('Closed Won / Implemented', 'Closed Lost.')
),

opp_max_stage_raw AS (
    SELECT
        ID AS OPPORTUNITY_ID,
        MAX(STAGE_NUMBER) AS MAX_STAGE_REACHED
    FROM DATALAB_SANDBOX.GS_REPORTING.OPTYS_STAGE_DURATION_WEIGHTED
    WHERE SLICE_START >= TRUE_CYCLE_START
      AND STAGE_NUMBER <= 5
    GROUP BY ID
),

backward_moves AS (
    SELECT
        ID AS OPPORTUNITY_ID,
        SUM(CASE WHEN IS_BACKWARD_MOVE THEN 1 ELSE 0 END) AS BACKWARD_MOVE_COUNT
    FROM DATALAB_SANDBOX.GS_REPORTING.OPTYS_STAGE_TRANSITIONS
    GROUP BY ID
),

base AS (
    SELECT
        ob.*,
        ms.MAX_STAGE_REACHED,
        COALESCE(bm.BACKWARD_MOVE_COUNT, 0) AS BACKWARD_MOVE_COUNT
    FROM opp_base ob
    JOIN opp_max_stage_raw ms ON ms.OPPORTUNITY_ID = ob.OPPORTUNITY_ID
    LEFT JOIN backward_moves bm ON bm.OPPORTUNITY_ID = ob.OPPORTUNITY_ID
),

-- ---------------------------------------------------------------------------
-- PRIOR ACCOUNT HISTORY (point-in-time safe)
-- ---------------------------------------------------------------------------
won_history AS (
    SELECT g.ID AS OPPORTUNITY_ID, g.ACCOUNTID, tc.TRUE_CLOSE_DATE
    FROM DATALAB_SANDBOX.GS_REPORTING.OPTYS_GOLD g
    JOIN true_close tc ON tc.OPPORTUNITY_ID = g.ID
    WHERE TRIM(tc.TRUE_CLOSE_STAGENAME) = 'Closed Won / Implemented'
),

prior_account_wins AS (
    SELECT
        b.OPPORTUNITY_ID,
        COUNT(w.OPPORTUNITY_ID) AS PRIOR_ACCOUNT_WIN_COUNT
    FROM base b
    LEFT JOIN won_history w
        ON w.ACCOUNTID = b.ACCOUNTID
       AND w.OPPORTUNITY_ID <> b.OPPORTUNITY_ID
       AND w.TRUE_CLOSE_DATE < b.TRUE_CYCLE_START
    GROUP BY b.OPPORTUNITY_ID
),

-- ---------------------------------------------------------------------------
-- ACTIVITY (account-level, per stage; completed tasks / attended meetings)
-- ---------------------------------------------------------------------------
stage_windows AS (
    SELECT d.ID AS OPPORTUNITY_ID, d.STAGE_NUMBER, d.SLICE_START, d.SLICE_END
    FROM DATALAB_SANDBOX.GS_REPORTING.OPTYS_STAGE_DURATION_WEIGHTED d
    WHERE d.ID IN (SELECT OPPORTUNITY_ID FROM base)
      AND d.SLICE_START >= d.TRUE_CYCLE_START
      AND d.STAGE_NUMBER <= 5
),

task_effective AS (
    SELECT ID, ACCOUNTID, SUBTYPE, COALESCE(ACTIVITYDATE, CREATEDDATE) AS EFFECTIVE_DATE
    FROM EDH.SFDC.TASK_V
    WHERE UPPER(STATUS) LIKE '%COMPLETED%'
      AND SUBTYPE IN ('Task', 'Cadence', 'ListEmail', 'Email', 'Call')
),

task_stage AS (
    SELECT
        b.OPPORTUNITY_ID,
        sw.STAGE_NUMBER,
        SUM(CASE WHEN t.SUBTYPE = 'Task' THEN 1 ELSE 0 END)      AS TASK_COUNT,
        SUM(CASE WHEN t.SUBTYPE = 'Cadence' THEN 1 ELSE 0 END)   AS CADENCE_COUNT,
        SUM(CASE WHEN t.SUBTYPE = 'ListEmail' THEN 1 ELSE 0 END) AS LISTEMAIL_COUNT,
        SUM(CASE WHEN t.SUBTYPE = 'Email' THEN 1 ELSE 0 END)     AS EMAIL_COUNT,
        SUM(CASE WHEN t.SUBTYPE = 'Call' THEN 1 ELSE 0 END)      AS CALL_COUNT
    FROM base b
    JOIN stage_windows sw ON sw.OPPORTUNITY_ID = b.OPPORTUNITY_ID
    LEFT JOIN task_effective t
        ON t.ACCOUNTID = b.ACCOUNTID
       AND t.EFFECTIVE_DATE BETWEEN sw.SLICE_START AND sw.SLICE_END
    GROUP BY b.OPPORTUNITY_ID, sw.STAGE_NUMBER
),

task_features AS (
    SELECT
        OPPORTUNITY_ID,
        SUM(IFF(STAGE_NUMBER = 1, TASK_COUNT, 0))      AS ACCT_TASK_COUNT_STAGE1,
        SUM(IFF(STAGE_NUMBER = 2, TASK_COUNT, 0))      AS ACCT_TASK_COUNT_STAGE2,
        SUM(IFF(STAGE_NUMBER = 3, TASK_COUNT, 0))      AS ACCT_TASK_COUNT_STAGE3,
        SUM(IFF(STAGE_NUMBER = 4, TASK_COUNT, 0))      AS ACCT_TASK_COUNT_STAGE4,
        SUM(IFF(STAGE_NUMBER = 5, TASK_COUNT, 0))      AS ACCT_TASK_COUNT_STAGE5,
        SUM(IFF(STAGE_NUMBER = 1, CADENCE_COUNT, 0))   AS ACCT_CADENCE_COUNT_STAGE1,
        SUM(IFF(STAGE_NUMBER = 2, CADENCE_COUNT, 0))   AS ACCT_CADENCE_COUNT_STAGE2,
        SUM(IFF(STAGE_NUMBER = 3, CADENCE_COUNT, 0))   AS ACCT_CADENCE_COUNT_STAGE3,
        SUM(IFF(STAGE_NUMBER = 4, CADENCE_COUNT, 0))   AS ACCT_CADENCE_COUNT_STAGE4,
        SUM(IFF(STAGE_NUMBER = 5, CADENCE_COUNT, 0))   AS ACCT_CADENCE_COUNT_STAGE5,
        SUM(IFF(STAGE_NUMBER = 1, LISTEMAIL_COUNT, 0)) AS ACCT_LISTEMAIL_COUNT_STAGE1,
        SUM(IFF(STAGE_NUMBER = 2, LISTEMAIL_COUNT, 0)) AS ACCT_LISTEMAIL_COUNT_STAGE2,
        SUM(IFF(STAGE_NUMBER = 3, LISTEMAIL_COUNT, 0)) AS ACCT_LISTEMAIL_COUNT_STAGE3,
        SUM(IFF(STAGE_NUMBER = 4, LISTEMAIL_COUNT, 0)) AS ACCT_LISTEMAIL_COUNT_STAGE4,
        SUM(IFF(STAGE_NUMBER = 5, LISTEMAIL_COUNT, 0)) AS ACCT_LISTEMAIL_COUNT_STAGE5,
        SUM(IFF(STAGE_NUMBER = 1, EMAIL_COUNT, 0))     AS ACCT_EMAIL_COUNT_STAGE1,
        SUM(IFF(STAGE_NUMBER = 2, EMAIL_COUNT, 0))     AS ACCT_EMAIL_COUNT_STAGE2,
        SUM(IFF(STAGE_NUMBER = 3, EMAIL_COUNT, 0))     AS ACCT_EMAIL_COUNT_STAGE3,
        SUM(IFF(STAGE_NUMBER = 4, EMAIL_COUNT, 0))     AS ACCT_EMAIL_COUNT_STAGE4,
        SUM(IFF(STAGE_NUMBER = 5, EMAIL_COUNT, 0))     AS ACCT_EMAIL_COUNT_STAGE5,
        SUM(IFF(STAGE_NUMBER = 1, CALL_COUNT, 0))      AS ACCT_CALL_COUNT_STAGE1,
        SUM(IFF(STAGE_NUMBER = 2, CALL_COUNT, 0))      AS ACCT_CALL_COUNT_STAGE2,
        SUM(IFF(STAGE_NUMBER = 3, CALL_COUNT, 0))      AS ACCT_CALL_COUNT_STAGE3,
        SUM(IFF(STAGE_NUMBER = 4, CALL_COUNT, 0))      AS ACCT_CALL_COUNT_STAGE4,
        SUM(IFF(STAGE_NUMBER = 5, CALL_COUNT, 0))      AS ACCT_CALL_COUNT_STAGE5
    FROM task_stage
    GROUP BY OPPORTUNITY_ID
),

event_effective AS (
    SELECT ID, ACCOUNTID, COALESCE(ACTIVITYDATE, CREATEDDATE) AS EFFECTIVE_DATE
    FROM EDH.SFDC.EVENT_V
    WHERE EVENT_STATUS = 'Attended'
),

event_stage AS (
    SELECT b.OPPORTUNITY_ID, sw.STAGE_NUMBER, COUNT(e.ID) AS MEET_COUNT
    FROM base b
    JOIN stage_windows sw ON sw.OPPORTUNITY_ID = b.OPPORTUNITY_ID
    LEFT JOIN event_effective e
        ON e.ACCOUNTID = b.ACCOUNTID
       AND e.EFFECTIVE_DATE BETWEEN sw.SLICE_START AND sw.SLICE_END
    GROUP BY b.OPPORTUNITY_ID, sw.STAGE_NUMBER
),

event_features AS (
    SELECT
        OPPORTUNITY_ID,
        SUM(IFF(STAGE_NUMBER = 1, MEET_COUNT, 0)) AS ACCT_MEET_COUNT_STAGE1,
        SUM(IFF(STAGE_NUMBER = 2, MEET_COUNT, 0)) AS ACCT_MEET_COUNT_STAGE2,
        SUM(IFF(STAGE_NUMBER = 3, MEET_COUNT, 0)) AS ACCT_MEET_COUNT_STAGE3,
        SUM(IFF(STAGE_NUMBER = 4, MEET_COUNT, 0)) AS ACCT_MEET_COUNT_STAGE4,
        SUM(IFF(STAGE_NUMBER = 5, MEET_COUNT, 0)) AS ACCT_MEET_COUNT_STAGE5
    FROM event_stage
    GROUP BY OPPORTUNITY_ID
),

-- ---------------------------------------------------------------------------
-- CONTACTS (current snapshot per account, active contacts only)
-- If ACTIVE__C is stored as text in EDH, change to ACTIVE__C = 'true'.
-- ---------------------------------------------------------------------------
contact_features AS (
    SELECT
        ACCOUNTID,
        COUNT(*) AS ACTIVE_CONTACT_COUNT,
        AVG(CASE WHEN (EMAIL IS NOT NULL OR PHONE IS NOT NULL)
                  AND COALESCE(DO_NOT_CONTACT__C, FALSE) = FALSE
                 THEN 1 ELSE 0 END) AS PCT_CONTACTS_CAN_CONTACT,
        SUM(IFF(JOB_CATEGORY__C = 'C-Suite', 1, 0)) AS JOBCAT_EXECUTIVE_COUNT,
        SUM(IFF(JOB_CATEGORY__C IN ('Finance', 'Procurement', 'Supply Chain'), 1, 0)) AS JOBCAT_FINANCE_PROCUREMENT_COUNT,
        SUM(IFF(JOB_CATEGORY__C IN ('Operations', 'Facilities', 'Admin'), 1, 0))      AS JOBCAT_OPERATIONS_COUNT,
        SUM(IFF(JOB_CATEGORY__C = 'IT', 1, 0))                                         AS JOBCAT_IT_COUNT,
        SUM(IFF(JOB_CATEGORY__C IN ('HR', 'Compliance'), 1, 0))                        AS JOBCAT_PEOPLE_GOVERNANCE_COUNT,
        SUM(IFF(JOB_CATEGORY__C = 'Marketing', 1, 0))                                  AS JOBCAT_MARKETING_COUNT,
        SUM(IFF(JOB_CATEGORY__C IN ('Nursing', 'School', 'Furniture'), 1, 0))          AS JOBCAT_VERTICAL_ROLE_COUNT,
        SUM(IFF(JOB_CATEGORY__C IS NULL OR JOB_CATEGORY__C NOT IN (
                'Marketing', 'IT', 'Finance', 'C-Suite', 'Operations', 'Compliance', 'Admin',
                'Supply Chain', 'Nursing', 'Facilities', 'Procurement', 'HR', 'Furniture', 'School'
            ), 1, 0)) AS JOBCAT_UNKNOWN_COUNT,
        MODE(LEADSOURCE) AS DOMINANT_LEADSOURCE
    FROM EDH.SFDC.CONTACT_V
    WHERE ACTIVE__C = TRUE
    GROUP BY ACCOUNTID
)

-- ---------------------------------------------------------------------------
-- FINAL — one row per open opportunity
-- ---------------------------------------------------------------------------
SELECT
    -- metadata / routing / gate (never model inputs)
    b.OPPORTUNITY_ID, b.ACCOUNTID, b.INITIATIVE__C, b.CURRENT_STAGENAME, b.OWNER_NAME,
    b.TRUE_CYCLE_START, b.DAYS_OPEN,
    b.OWNER_IS_ACTIVE, b.OWNER_IS_NON_HUMAN, b.IS_PROSPECT_DESK, b.IS_DEAD_QUEUE,
    -- static features (both models)
    b.VERTICAL, b.SECTOR, b.HEADCOUNT, b.RECORDTYPE_NAME, b.BILLINGSTATE, b.IS_REMOTE_GEO,
    COALESCE(paw.PRIOR_ACCOUNT_WIN_COUNT, 0) AS PRIOR_ACCOUNT_WIN_COUNT,
    cf.ACTIVE_CONTACT_COUNT, cf.PCT_CONTACTS_CAN_CONTACT, cf.DOMINANT_LEADSOURCE,
    cf.JOBCAT_EXECUTIVE_COUNT, cf.JOBCAT_FINANCE_PROCUREMENT_COUNT, cf.JOBCAT_OPERATIONS_COUNT,
    cf.JOBCAT_IT_COUNT, cf.JOBCAT_PEOPLE_GOVERNANCE_COUNT, cf.JOBCAT_MARKETING_COUNT,
    cf.JOBCAT_VERTICAL_ROLE_COUNT, cf.JOBCAT_UNKNOWN_COUNT,
    -- Prospect+ only
    b.AMOUNT, b.AMOUNT_BUCKET, b.MAX_STAGE_REACHED, b.BACKWARD_MOVE_COUNT, b.CYCLE_LENGTH_DAYS,
    tf.* EXCLUDE (OPPORTUNITY_ID),
    ef.* EXCLUDE (OPPORTUNITY_ID)
FROM base b
LEFT JOIN prior_account_wins paw ON paw.OPPORTUNITY_ID = b.OPPORTUNITY_ID
LEFT JOIN task_features tf       ON tf.OPPORTUNITY_ID = b.OPPORTUNITY_ID
LEFT JOIN event_features ef      ON ef.OPPORTUNITY_ID = b.OPPORTUNITY_ID
LEFT JOIN contact_features cf    ON cf.ACCOUNTID = b.ACCOUNTID
;
