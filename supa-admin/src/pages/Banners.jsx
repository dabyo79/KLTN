import { useEffect, useState } from 'react'
import { supabase } from '../supabaseClient'

const BUCKET = 'laptopbanchon'

export default function Banners() {
  const [list, setList] = useState([])
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')

  const load = async () => {
    const { data } = await supabase.from('banners').select('*').order('created_at', { ascending: false })
    setList(data ?? [])
  }

  useEffect(() => {
    load()
  }, [])

  const handleAdd = async (e) => {
    e.preventDefault()
    let imageUrl = ''
    if (file) {
      const ext = file.name.split('.').pop()
      const path = `banners/${crypto.randomUUID()}.${ext}`
      const { error } = await supabase.storage.from(BUCKET).upload(path, file, { upsert: true })
      if (error) {
        alert(error.message)
        return
      }
      imageUrl = supabase.storage.from(BUCKET).getPublicUrl(path).data.publicUrl
    }
    const { error } = await supabase.from('banners').insert({ title, imageUrl })
    if (error) alert(error.message)
    else {
      setTitle('')
      setFile(null)
      load()
    }
  }

  const handleDelete = async (id) => {
    if (!confirm('Xoá banner này?')) return
    await supabase.from('banners').delete().eq('id', id)
    load()
  }

  return (
    <div>
      <h4>🖼 Banners</h4>
      <form className="row g-3 mb-3" onSubmit={handleAdd}>
        <div className="col-md-4">
          <input
            className="form-control"
            placeholder="Tiêu đề"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
        </div>
        <div className="col-md-4">
          <input
            className="form-control"
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </div>
        <div className="col-md-2">
          <button className="btn btn-primary">Thêm</button>
        </div>
      </form>

      <div className="row g-3">
        {list.map((b) => (
          <div className="col-md-3" key={b.id}>
            <div className="card">
              {b.imageUrl && (
                <img src={b.imageUrl} className="card-img-top" style={{ height: 120, objectFit: 'cover' }} />
              )}
              <div className="card-body d-flex justify-content-between">
                <div>{b.title}</div>
                <button className="btn btn-sm btn-outline-danger" onClick={() => handleDelete(b.id)}>
                  Xoá
                </button>
              </div>
            </div>
          </div>
        ))}
        {list.length === 0 && <p className="text-muted">Chưa có banner</p>}
      </div>
    </div>
  )
}
