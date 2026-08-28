-- Verifică dacă există DOUĂ conturi "sergiu" (unul @citim.app, unul @test.com)
select id as user_id, email, created_at
from auth.users
where lower(email) like 'sergiu@%'
order by created_at;
