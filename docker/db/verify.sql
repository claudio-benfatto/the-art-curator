-- Proves the image can do what the schema needs: both extensions load and their operators work.
-- Run with `psql -v ON_ERROR_STOP=1 -f docker/db/verify.sql`. Used by the db-image workflow.
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

SELECT extname, extversion FROM pg_extension WHERE extname IN ('postgis', 'vector') ORDER BY 1;

-- MACBA -> Fundació Joan Miró, metres on the geography type (~1745 m).
SELECT round(ST_Distance(
    ST_MakePoint(2.1667, 41.3833)::geography,
    ST_MakePoint(2.1597, 41.3685)::geography
)) AS macba_to_miro_m;

SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector AS cosine_distance;
