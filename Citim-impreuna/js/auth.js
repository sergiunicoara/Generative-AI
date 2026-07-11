/* Autentificare Supabase Auth.
   Username-ul este mapat la `username@citim.app` în câmpul email — utilizatorul
   vede doar câmpul "Nume utilizator", niciodată adresa de email internă.

   SETUP NECESAR în Supabase Dashboard:
     Authentication → Settings → dezactivează „Enable email confirmations"
*/

const Auth = (() => {
  const DOMAIN = '@test.com';
  let _client = null;
  let _session = null;
  let _onChangeCb = null;

  function client() {
    if (!_client && typeof supabase !== 'undefined') {
      _client = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
        auth: { persistSession: true, storageKey: 'ci_auth' },
      });
    }
    return _client;
  }

  function toEmail(username) {
    return username.trim().toLowerCase().replace(/\s+/g, '_') + DOMAIN;
  }

  function extractUsername(user) {
    if (!user) return null;
    const raw = user.user_metadata?.username ||
                (user.email || '').replace(DOMAIN, '') ||
                null;
    if (!raw) return null;
    return raw.replace(/\b\w/g, c => c.toUpperCase());
  }

  function translateError(msg) {
    if (!msg) return 'Eroare necunoscută.';
    if (msg.includes('Invalid login') || msg.includes('invalid_credentials')) return 'Nume sau parolă incorectă.';
    if (msg.includes('already registered') || msg.includes('already been registered')) return 'Acest nume de utilizator este deja folosit.';
    if (msg.includes('Password should')) return 'Parola trebuie să aibă cel puțin 6 caractere.';
    if (msg.includes('rate limit') || msg.includes('over_email_send_rate_limit')) return 'Prea multe încercări. Încearcă mai târziu.';
    if (msg.includes('Email not confirmed')) return 'Confirmarea prin email este activată — contactează administratorul.';
    return msg;
  }

  async function init(onChangeCb) {
    _onChangeCb = onChangeCb;
    const c = client();
    if (!c) return null; // SDK indisponibil (offline la prima încărcare)
    const { data } = await c.auth.getSession();
    _session = data.session;
    c.auth.onAuthStateChange((_, session) => {
      _session = session;
      if (_onChangeCb) _onChangeCb(currentUser());
    });
    return currentUser();
  }

  async function signIn(username, password) {
    if (!client()) throw new Error('Serviciul de autentificare nu este disponibil. Verifică conexiunea la internet.');
    const { data, error } = await client().auth.signInWithPassword({
      email: toEmail(username),
      password,
    });
    if (error) throw new Error(translateError(error.message));
    _session = data.session;
    return currentUser();
  }

  async function signUp(username, password) {
    if (!client()) throw new Error('Serviciul de autentificare nu este disponibil. Verifică conexiunea la internet.');
    const trimmed = username.trim();
    if (trimmed.length < 2) throw new Error('Numele trebuie să aibă cel puțin 2 caractere.');
    if (password.length < 6) throw new Error('Parola trebuie să aibă cel puțin 6 caractere.');
    const { data, error } = await client().auth.signUp({
      email: toEmail(trimmed),
      password,
      options: { data: { username: trimmed } },
    });
    if (error) throw new Error(translateError(error.message));
    if (data.user && !data.session) {
      throw new Error('Confirmarea prin email este activată în Supabase — dezactiveaz-o din Dashboard → Auth → Settings.');
    }
    _session = data.session;
    return currentUser();
  }

  async function signOut() {
    if (client()) await client().auth.signOut();
    _session = null;
  }

  function currentUser() {
    if (!_session) return null;
    return extractUsername(_session.user);
  }

  function isLoggedIn() { return !!_session; }

  return { init, signIn, signUp, signOut, currentUser, isLoggedIn };
})();
