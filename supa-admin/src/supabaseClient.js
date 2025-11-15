import { createClient } from '@supabase/supabase-js'

// TODO: thay bằng của bạn
const supabaseUrl = 'https://korlofxtailwltuhydya.supabase.co'
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtvcmxvZnh0YWlsd2x0dWh5ZHlhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI0OTE4NTEsImV4cCI6MjA3ODA2Nzg1MX0.Z0obqdlv31ce66ks6dCpZzEDLGLQ1D0A3QcltowP9xc'

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
