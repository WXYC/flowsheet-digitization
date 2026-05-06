-- build_wxyc_track_mv.sql — one-shot materialized view scoping the Discogs
-- (artist, track) cross product to releases credited to WXYC-owned artists.
--
-- Why: querying the full release_track table (174M rows) by trigram on title
-- takes ~700ms cold per query, even with the trigram index. Pre-filtering to
-- the WXYC artist subset (26K artists from wxyc_library_artist) reduces the
-- target table to a few-hundred-thousand rows, making (artist, track) lookups
-- sub-millisecond.
--
-- Used by: scripts/spot_check_discogs.py (and any future phase-2 reconciliation).
--
-- Run once:
--     psql -h localhost -p 5432 -d discogs -f scripts/build_wxyc_track_mv.sql
--
-- Refresh after a Discogs cache rebuild:
--     REFRESH MATERIALIZED VIEW wxyc_track;

CREATE MATERIALIZED VIEW IF NOT EXISTS wxyc_track AS
SELECT DISTINCT
    lower(f_unaccent(ra.artist_name)) AS artist_norm,
    lower(f_unaccent(rt.title))       AS track_norm
FROM release_artist ra
JOIN release_track rt USING (release_id)
JOIN wxyc_library_artist wla
    ON lower(f_unaccent(ra.artist_name)) = wla.norm_name
WHERE ra.extra = 0;

CREATE UNIQUE INDEX IF NOT EXISTS idx_wxyc_track_pair
    ON wxyc_track(artist_norm, track_norm);

CREATE INDEX IF NOT EXISTS idx_wxyc_track_artist
    ON wxyc_track(artist_norm);
