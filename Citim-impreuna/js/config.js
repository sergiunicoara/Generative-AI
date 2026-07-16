// Configurare Supabase pentru statistici de activitate.
// Cheia "anon" este publică prin design — protecția reală vine din politicile
// RLS (Row Level Security) de pe tabelul "events".
// Lăsați ambele goale pentru a dezactiva complet trimiterea statisticilor
// (aplicația funcționează normal, doar fără urmărirea activității).

const SUPABASE_URL = "https://szwrfxcshcbqgdtfqqfp.supabase.co";
const SUPABASE_ANON_KEY = "sb_publishable_n1Lq2iYE2_vrWnHiZXQuzw_Us2HFr3c";
