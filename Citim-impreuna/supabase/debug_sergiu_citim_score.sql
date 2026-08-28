-- Diagnostic: de ce nu apare scorul lui sergiu@citim.app în clasament.
-- Rulează pas cu pas în Supabase SQL Editor (fiecare select separat).

-- 1) Există contul, și care e user_id-ul lui?
select id as user_id, email, created_at
from auth.users
where lower(email) = 'sergiu@citim.app';

-- 2) Are evenimente înregistrate (adică a răspuns vreodată la ceva)?
select count(*) as total_events,
       count(*) filter (where correct is true) as correct_events,
       max(created_at) as last_event
from public.events
where user_id = (select id from auth.users where lower(email) = 'sergiu@citim.app');

-- 3) Are deja un rând în scores?
select *
from public.scores
where user_id = (select id from auth.users where lower(email) = 'sergiu@citim.app');

-- 4) Dacă (2) arată evenimente dar (3) e gol sau are 0 puncte greșit,
--    forțează recalcularea manual (funcția există din 20260813_server_score_trigger.sql):
-- select public.recalculate_score_for_user(
--   (select id from auth.users where lower(email) = 'sergiu@citim.app')
-- );

-- 5) Verifică apoi din nou (3), și separat că funcția publică chiar îl întoarce
--    (rulează ca owner/service role ca să ocolești filtrul pe domeniu din JWT):
-- select * from public.get_public_leaderboard(1000)
-- where user_id = (select id from auth.users where lower(email) = 'sergiu@citim.app');
