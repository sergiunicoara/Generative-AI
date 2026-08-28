-- Arată definiția LIVE a funcției get_public_leaderboard din Supabase.
-- Dacă în rezultat NU apare "case when lower(auth.jwt() ->> 'email') like '%@citim.app'",
-- migrația 20260824_leaderboard_domain_split.sql nu a fost încă rulată — trebuie
-- aplicată (copiază tot conținutul fișierului în SQL Editor și rulează-l).
select pg_get_functiondef('public.get_public_leaderboard'::regproc);
