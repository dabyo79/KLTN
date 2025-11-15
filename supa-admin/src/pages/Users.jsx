import { useEffect, useState } from 'react'
import { supabase } from '../supabaseClient'
import { Doughnut } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from 'chart.js'

ChartJS.register(ArcElement, Tooltip, Legend)

const COLORS = [
  '#ff6384',
  '#36a2eb',
  '#ffcd56',
  '#4bc0c0',
  '#9966ff',
  '#ff9f40',
]

export default function Users() {
  const [users, setUsers] = useState([])
  const [search, setSearch] = useState('')
  const [selectedUser, setSelectedUser] = useState(null)
  const [userStats, setUserStats] = useState(null)
  const [loadingStats, setLoadingStats] = useState(false)

  useEffect(() => {
    const load = async () => {
      const { data, error } = await supabase
        .from('profiles')
        .select('id, full_name, email, role, avatar_url')
        .order('created_at', { ascending: false })
      if (!error) setUsers(data ?? [])
    }
    load()
  }, [])

  const filtered = users.filter((u) => {
    const k = search.toLowerCase()
    return (
      u.full_name?.toLowerCase().includes(k) ||
      u.email?.toLowerCase().includes(k) ||
      u.role?.toLowerCase().includes(k)
    )
  })

  // gọi flask
  const loadUserStats = async (user) => {
  setSelectedUser(user)  // lưu cả tên, avatar...
  setLoadingStats(true)
  try {
    const res = await fetch(`http://127.0.0.1:5000/admin/api/user_stats?user_id=${user.id}`)
    const data = await res.json()
    setUserStats(data)
  } catch (e) {
    console.error(e)
    setUserStats(null)
  } finally {
    setLoadingStats(false)
  }
}


  // dữ liệu từ API
  const recentRec = userStats?.recent_recommends || []        // 5 query gần nhất
  const topClicked = userStats?.top_clicked || []             // top 10 laptop click
  const longestStay = userStats?.longest_stay || []           // laptop xem lâu nhất
  const priceBuckets = userStats?.price_buckets || {}         // để vẽ doughnut

  // doughnut phân khúc giá
  const priceLabels = Object.keys(priceBuckets)
  const chartPrice = {
    labels: priceLabels,
    datasets: [
      {
        data: priceLabels.map((k) => priceBuckets[k]),
        backgroundColor: priceLabels.map((_, i) => COLORS[i % COLORS.length]),
      },
    ],
  }

  return (
    <div>
      <h4>👤 Người dùng</h4>
      <div className="input-group mb-3 mt-3">
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          type="text"
          className="form-control"
          placeholder="Tìm theo tên hoặc email..."
        />
        <span className="input-group-text">
          <i className="fas fa-search"></i>
        </span>
      </div>
      <div className="table-responsive">
        <table className="table table-hover align-middle">
          <thead>
            <tr>
              <th>Ảnh</th>
              <th>Tên</th>
              <th>Email</th>
              <th>Vai trò</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((u) => (
              <tr key={u.id}>
                <td>
                  <img
                    src={u.avatar_url || 'https://placehold.co/40x40'}
                    alt=""
                    width="40"
                    height="40"
                    style={{ borderRadius: '50%', objectFit: 'cover' }}
                  />
                </td>
                <td>{u.full_name || '—'}</td>
                <td>{u.email}</td>
                <td>
                  <span
                    className={`badge bg-${u.role === 'admin' ? 'danger' : 'secondary'}`}
                  >
                    {u.role || 'user'}
                  </span>
                </td>
                <td>
                  <button
                    className="btn btn-sm btn-outline-primary"
                    onClick={() => loadUserStats(u)}
                  >
                    Xem thống kê
                  </button>
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan="5" className="text-center text-muted">
                  Không có người dùng
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* panel thống kê */}
      {selectedUser && (
        <div className="mt-4">
          <h5>Thống kê cho user: {selectedUser.full_name || selectedUser.email || selectedUser.id}</h5>
          {loadingStats ? (
            <p>Đang tải...</p>
          ) : !userStats ? (
            <p>Không lấy được dữ liệu.</p>
          ) : (
            <div className="row">
              {/* 5 lượt dùng gợi ý gần đây → LIST */}
              <div className="col-md-6 mb-4">
  <div className="card p-3 h-100">
    <h6>5 lượt dùng gợi ý gần đây</h6>
    {recentRec.length === 0 ? (
      <p className="text-muted mb-0">Chưa có dữ liệu.</p>
    ) : (
      <ul className="list-group list-group-flush">
        {recentRec.map((r) => {
          const raw = (r.raw_query || r.query || '').trim()

          // parse struct
          let parsed = r.parsed_struct
          if (parsed && typeof parsed === 'string') {
            try {
              parsed = JSON.parse(parsed)
            } catch {
              parsed = null
            }
          }

          // chỉ gom các filter do người dùng bấm
          const realFilterParts = []
          if (parsed && typeof parsed === 'object') {
            if (parsed.filter_brand) {
              realFilterParts.push(`thương hiệu=${parsed.filter_brand}`)
            }
            if (parsed.min_price) {
              realFilterParts.push(
                `giá từ ${Number(parsed.min_price).toLocaleString('vi-VN')}đ`
              )
            }
            if (parsed.max_price) {
              realFilterParts.push(
                `giá đến ${Number(parsed.max_price).toLocaleString('vi-VN')}đ`
              )
            }
          }

          let line = ''
          if (raw) {
            // user có gõ chữ → ưu tiên in đúng chữ
            line = raw
            // chỉ nối thêm nếu thật sự có filter người dùng chọn
            if (realFilterParts.length > 0) {
              line += ' + lọc: ' + realFilterParts.join(', ')
            }
          } else {
            // user không gõ chữ → lúc này có thể hiển thị cả intent
            const intentParts = []
            if (parsed && typeof parsed === 'object') {
              if (parsed.brand) intentParts.push(`thương hiệu=${parsed.brand}`)
              if (parsed.budget) {
                intentParts.push(
                  `giá≤${Number(parsed.budget).toLocaleString('vi-VN')}đ`
                )
              }
              if (parsed.usage && Array.isArray(parsed.usage) && parsed.usage.length > 0) {
                intentParts.push(`mục đích=${parsed.usage.join(', ')}`)
              }
            }
            if (realFilterParts.length > 0) {
              intentParts.push(...realFilterParts)
            }

            line = intentParts.length > 0
              ? 'Lọc: ' + intentParts.join(', ')
              : '(tìm tất cả)'
          }

          return (
            <li key={r.id || r.created_at} className="list-group-item">
              <div className="fw-semibold">{line}</div>
              <small className="text-muted">
                {r.created_at ? new Date(r.created_at).toLocaleString() : ''}
              </small>
            </li>
          )
        })}
      </ul>
    )}
  </div>
</div>


              {/* 10 laptop click gần nhất → LIST */}
              <div className="col-md-6 mb-4">
                <div className="card p-3 h-100">
                  <h6>10 laptop được click xem nhiều nhất</h6>
                  {topClicked.length === 0 ? (
                    <p className="text-muted mb-0">Chưa có click nào.</p>
                  ) : (
                    <ul className="list-group list-group-flush">
                      {topClicked.map((c) => (
                        <li key={c.laptop_id} className="list-group-item d-flex align-items-center gap-2">
                          {c.image_url ? (
                            <img
                              src={c.image_url}
                              alt={c.name}
                              style={{ width: 40, height: 40, objectFit: 'cover', borderRadius: 6 }}
                            />
                          ) : (
                            <div style={{ width: 40, height: 40, background: '#eee', borderRadius: 6 }} />
                          )}
                          <div className="flex-grow-1">
                            <div className="fw-semibold">{c.name}</div>
                            <small className="text-muted">
                              {c.price ? c.price.toLocaleString('vi-VN') + ' ₫' : '—'}
                            </small>
                          </div>
                          <span className="badge bg-primary rounded-pill">
                            {c.total} lần
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>

              {/* laptop xem lâu nhất → LIST */}
              <div className="col-md-6 mb-4">
  <div className="card p-3 h-100">
    <h6>Các laptop user dừng lại lâu nhất</h6>
    {longestStay.length === 0 ? (
      <p className="text-muted mb-0">Chưa đo được thời gian xem.</p>
    ) : (
      <ul className="list-group list-group-flush">
        {longestStay.map((item) => {
          // format ms -> m:s
          const ms = item.duration_ms || 0
          const seconds = Math.floor(ms / 1000)
          const minutes = Math.floor(seconds / 60)
          const remainSeconds = seconds % 60
          const timeText =
            minutes > 0
              ? `${minutes}p ${remainSeconds}s`
              : `${remainSeconds}s`

          return (
            <li
              key={item.laptop_id}
              className="list-group-item d-flex align-items-center gap-2"
            >
              {item.image_url ? (
                <img
                  src={item.image_url}
                  alt={item.name}
                  style={{
                    width: 40,
                    height: 40,
                    objectFit: 'cover',
                    borderRadius: 6,
                  }}
                />
              ) : (
                <div
                  style={{
                    width: 40,
                    height: 40,
                    background: '#eee',
                    borderRadius: 6,
                  }}
                />
              )}
              <div className="flex-grow-1">
                <div className="fw-semibold">
                  {item.name || 'Không rõ tên'}
                </div>
                {item.created_at && (
                  <small className="text-muted">
                    {new Date(item.created_at).toLocaleString()}
                  </small>
                )}
              </div>
              <span className="badge bg-success rounded-pill">
                {timeText}
              </span>
            </li>
          )
        })}
      </ul>
    )}
  </div>
</div>

              {/* giỏ hàng của user */}
              <div className="col-md-6 mb-4">
                <div className="card p-3 h-100">
                  <h6>Sản phẩm trong giỏ</h6>
                  {userStats.carts && userStats.carts.length > 0 ? (
                    <ul className="list-group list-group-flush">
                      {userStats.carts.map((c) => (
                        <li key={c.id} className="list-group-item d-flex justify-content-between">
                          <span>{c.laptop_name || c.laptop_id}</span>
                          <span>x {c.quantity || 1}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-muted mb-0">Chưa có gì trong giỏ.</p>
                  )}
                </div>
              </div>

              {/* phân khúc giá */}
              <div className="col-md-6 mb-4">
                <div className="card p-3 h-100">
                  <h6>Phân khúc giá user hay xem</h6>
                  {priceLabels.length === 0 ? (
                    <p className="text-muted mb-0">Chưa có dữ liệu giá.</p>
                  ) : (
                    <Doughnut data={chartPrice} />
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
