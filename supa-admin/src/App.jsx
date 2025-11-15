import { useEffect, useState } from 'react'
import Sidebar from './components/Sidebar'
import Topbar from './components/Topbar'
import Dashboard from './pages/Dashboard'
import Users from './pages/Users'
import Products from './pages/Products'
import Banners from './pages/Banners'
import { supabase } from './supabaseClient'

function App() {
  const [activeTab, setActiveTab] = useState(
    () => localStorage.getItem('admin-active-tab') || 'dashboard'
  )
  const [adminInfo, setAdminInfo] = useState(null)

  useEffect(() => {
    localStorage.setItem('admin-active-tab', activeTab)
  }, [activeTab])

  // login kiểu “đang dùng app Android rồi copy token sang đây” thì bạn có thể bỏ phần này
  // tạm thời: fetch user hiện tại bằng supabase.auth.getUser()
  useEffect(() => {
    const loadUser = async () => {
      const { data } = await supabase.auth.getUser()
      setAdminInfo(data?.user ?? null)
    }
    loadUser()
  }, [])

  return (
    <div className="d-flex">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} adminInfo={adminInfo} />
      <div className="flex-grow-1" style={{ marginLeft: 250 }}>
        <Topbar />
        
        <div
    className="p-3"
    style={{
      background: '#f5f5f5',   // màu xám nhạt
      minHeight: '100vh'       // để kéo từ trên xuống dưới
    }}
  >
          {activeTab === 'dashboard' && <Dashboard />}
          {activeTab === 'users' && <Users />}
          {activeTab === 'products' && <Products />}
          {activeTab === 'banners' && <Banners />}
          {/* các tab khác bạn thêm tương tự */}
        </div>
        
      </div>

    </div>
  );
}


export default App
