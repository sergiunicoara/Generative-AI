-- Rulează din nou, ACUM, după ce ai completat pagina fiind logat pe sergiu@citim.app.

-- 1) Au ajuns evenimentele pe server?
select count(*) as total_events, max(created_at) as last_event
from public.events
where user_id = (select id from auth.users where lower(email) = 'sergiu@citim.app');

-- 2) S-a creat/actualizat rândul de scor?
select * from public.scores
where user_id = (select id from auth.users where lower(email) = 'sergiu@citim.app');
