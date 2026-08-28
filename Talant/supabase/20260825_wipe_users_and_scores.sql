-- Talant: golire completă a userilor și a istoricului de scoruri/încercări.
-- ATENȚIE: ireversibil. Nu există undo — nu rula fără un backup dacă ai
-- vreun dubiu. Rulează manual, o singură dată, în SQL Editor-ul proiectului
-- Supabase (nu e inclus în vreun flow automat).
--
-- Șterge:
--   - toți userii (auth.users) — orice cont, real sau de test
--   - tot istoricul quiz-ului Ioan (talant_attempts, talant_scores)
--   - tot istoricul testelor 1 Samuel (talant_test_attempts, talant_test_scores)
--
-- NU atinge tabelele de configurare/conținut (talant_quiz_answer_keys,
-- talant_test_answer_keys, talant_church_domains) — nu sunt date de user.

begin;

-- Golește explicit tabelele aplicației (redundant cu ON DELETE CASCADE de pe
-- user_id, dar explicit, ca să fie clar ce se șterge).
truncate table
  public.talant_attempts,
  public.talant_scores,
  public.talant_test_attempts,
  public.talant_test_scores
cascade;

-- Șterge toți userii din Supabase Auth. Cascadă automată către orice rând
-- rămas legat de user_id (deja golit mai sus) + tabelele interne auth.* ale
-- Supabase (identities, sessions etc., care au FK cu ON DELETE CASCADE pe
-- auth.users).
delete from auth.users;

commit;
