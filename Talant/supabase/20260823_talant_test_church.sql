-- Talant: grupare automată a clasamentului testelor pe baza domeniului de
-- email al contului — fără niciun câmp nou în formular, fără modificări în
-- cod. Conturile create cu un email dintr-un domeniu din
-- talant_church_domains (ex. xxx@test.com) intră în grupa "biserica";
-- restul (conturi normale, sintetice @talant.app sau orice alt domeniu)
-- intră în grupa "general". Clasamentul (talant_test_leaderboard) rămâne cu
-- aceeași semnătură ca înainte — se filtrează automat pe grupa userului
-- care întreabă, deci fiecare grupă își vede doar propriul clasament.
-- Rulează o singură dată în SQL Editor-ul proiectului Supabase, după
-- migrarea 20260822_talant_test_scoring.sql.
begin;

create table if not exists public.talant_church_domains (
  domain text primary key
);
insert into public.talant_church_domains (domain) values ('test.com')
  on conflict (domain) do nothing;

alter table public.talant_test_scores add column if not exists group_name text not null default 'general';
create index if not exists talant_test_scores_group_idx on public.talant_test_scores (quiz_version, group_name);

-- Determină grupa contului curent din domeniul emailului din JWT.
create or replace function public.talant_user_group()
returns text language sql stable security definer set search_path = public, pg_temp as $$
  select case
    when exists (
      select 1 from public.talant_church_domains d
      where lower(auth.jwt() ->> 'email') like ('%@' || lower(d.domain))
    ) then 'biserica'
    else 'general'
  end;
$$;

-- Recalculează scorul propriu, acum inclusiv grupa (biserica/general).
create or replace function public.talant_test_recalculate_own_score(p_quiz_version text)
returns void language plpgsql security definer set search_path = public, pg_temp as $$
declare
  v_user_id uuid := auth.uid();
  v_name text;
  v_group text;
  v_best integer;
  v_max integer;
  v_attempts integer;
begin
  if v_user_id is null then raise exception 'Authentication is required'; end if;
  v_name := initcap(coalesce(nullif(trim(auth.jwt() -> 'user_metadata' ->> 'username'), ''), 'Utilizator'));
  v_group := public.talant_user_group();
  select max(total_points), max(max_points), count(*)
    into v_best, v_max, v_attempts
    from public.talant_test_attempts
    where user_id = v_user_id and quiz_version = p_quiz_version;
  insert into public.talant_test_scores (user_id, quiz_version, user_name, group_name, best_points, max_points, attempts, updated_at)
  values (v_user_id, p_quiz_version, v_name, v_group, coalesce(v_best, 0), coalesce(v_max, 0), coalesce(v_attempts, 0), now())
  on conflict (user_id, quiz_version) do update set
    user_name = excluded.user_name, group_name = excluded.group_name, best_points = excluded.best_points,
    max_points = excluded.max_points, attempts = excluded.attempts, updated_at = excluded.updated_at;
end;
$$;

-- Clasament — semnătură neschimbată (p_quiz_version, p_limit); filtrează
-- automat pe grupa userului care întreabă, deci fiecare grupă e izolată.
create or replace function public.talant_test_leaderboard(p_quiz_version text, p_limit integer default 20)
returns table(rank bigint, user_name text, best_points integer, max_points integer)
language sql security definer set search_path = public, pg_temp as $$
  select row_number() over (order by s.best_points desc, s.updated_at asc), s.user_name, s.best_points, s.max_points
  from public.talant_test_scores s
  where s.quiz_version = p_quiz_version and s.best_points > 0
    and s.group_name = public.talant_user_group()
  order by s.best_points desc, s.updated_at asc limit least(greatest(p_limit, 1), 100);
$$;

-- Backfill: recalculează grupa pentru scorurile deja existente.
update public.talant_test_scores s
set group_name = case
  when exists (
    select 1 from public.talant_church_domains d, auth.users u
    where u.id = s.user_id and lower(u.email) like ('%@' || lower(d.domain))
  ) then 'biserica'
  else 'general'
end;

revoke all on function public.talant_user_group() from public;
revoke all on function public.talant_test_leaderboard(text, integer) from public;
grant execute on function public.talant_user_group() to authenticated;
grant execute on function public.talant_test_leaderboard(text, integer) to authenticated;

commit;
