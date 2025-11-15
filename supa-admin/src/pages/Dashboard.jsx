import { useEffect, useState } from "react";
import { supabase } from "../supabaseClient";
import { Bar, Line } from "react-chartjs-2";
import {

  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Tooltip,
  Legend,
  Filler
);


ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

// bảng màu dùng chung
const BAR_COLORS = [
  "rgba(255, 99, 132, 0.7)",
  "rgba(54, 162, 235, 0.7)",
  "rgba(255, 206, 86, 0.7)",
  "rgba(75, 192, 192, 0.7)",
  "rgba(153, 102, 255, 0.7)",
  "rgba(255, 159, 64, 0.7)",
  "rgba(255, 99, 255, 0.7)",
  "rgba(99, 255, 132, 0.7)",
  "rgba(132, 99, 255, 0.7)",
  "rgba(99, 132, 255, 0.7)",
];

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    users: 0,
    laptops: 0,
    banners: 0,
  });

  const [flaskData, setFlaskData] = useState(null);
  const [flaskLoading, setFlaskLoading] = useState(true);

  // ===== SUPABASE COUNTERS =====
  useEffect(() => {
    const load = async () => {
      setLoading(true);

      const { count: userCount } = await supabase
        .from("profiles")
        .select("*", { count: "exact", head: true });

      const { count: laptopCount } = await supabase
        .from("laptops")
        .select("*", { count: "exact", head: true });

      const { count: bannerCount, error: bannerErr } = await supabase
        .from("banners")
        .select("*", { count: "exact", head: true });

      setStats({
        users: userCount ?? 0,
        laptops: laptopCount ?? 0,
        banners: bannerErr ? 0 : bannerCount ?? 0,
      });

      setLoading(false);
    };

    load();
  }, []);

  // ===== FLASK LOGS =====
  useEffect(() => {
    const loadFlask = async () => {
      try {
        setFlaskLoading(true);
        const res = await fetch("http://127.0.0.1:5000/admin/api/stats_json");
        const data = await res.json();
        setFlaskData(data);
      } catch (err) {
        console.error("Không lấy được stats từ Flask:", err);
        setFlaskData(null);
      } finally {
        setFlaskLoading(false);
      }
    };

    loadFlask();
  }, []);

  // ====== TÁCH DỮ LIỆU RA ======
  const logs = flaskData?.logs || [];
  const trafficLogs = flaskData?.traffic_logs || [];

  
  const topSearchUsers = (flaskData?.top_search_users || []).slice(0, 20);
  // sort brand từ nhiều → ít
  const brandFromLogs = (flaskData?.brand_from_logs || []).slice().sort(
    (a, b) => (b.total || 0) - (a.total || 0)
  );
  const topClicked = flaskData?.top_clicked_laptops || [];
  const topCart = flaskData?.top_cart_laptops || [];

  // ====== 1. chart “lượt gợi ý theo ngày” ======
  // ====== 1. Lượt gợi ý từ Flask (5 ngày gần nhất, ngang) ======
const searchesByDate = {};
logs.forEach((item) => {
  const d = item.created_at ? item.created_at.slice(0, 10) : "không rõ";
  searchesByDate[d] = (searchesByDate[d] || 0) + 1;
});

// lấy 5 ngày gần nhất
const sortedDates = Object.keys(searchesByDate).sort().reverse(); // mới nhất trước
const latest5 = sortedDates.slice(0, 5).reverse(); // đảo lại để hiển thị cũ -> mới

const flaskHorizontalData = {
  labels: latest5,
  datasets: [
    {
      label: "Lượt gợi ý",
      data: latest5.map((d) => searchesByDate[d]),
      backgroundColor: "rgba(54, 162, 235, 0.7)",
      borderColor: "rgba(54, 162, 235, 1)",
      borderWidth: 1,
      borderRadius: 16,          // bo cong giống hình
      barThickness: 22,          // mỏng mỏng
    },
  ],
};

const flaskHorizontalOptions = {
  indexAxis: "y",                // 👈 chuyển sang ngang
  responsive: true,
  plugins: {
    legend: {
      display: false,            // bỏ ô legend
    },
    tooltip: {
      enabled: true,
    },
  },
  scales: {
    x: {
      beginAtZero: true,
      grid: {
        display: true,
      },
      ticks: {
        stepSize: 1,
      },
    },
    y: {
      grid: {
        display: false,
      },
    },
  },
};


  const commonNoLegend = {
    responsive: true,
    plugins: {
      legend: { display: false },
    },
    scales: {
      y: { beginAtZero: true },
    },
  };

  // ===== traffic: đếm theo device từ logs =====
  // ===== traffic theo giờ hôm nay =====
// ===== traffic theo giờ hôm nay =====
const trafficToday = (() => {
  if (!trafficLogs.length) return { labels: [], counts: [] };

  const now = new Date();
  const yyyy = now.getFullYear();
  const mm = String(now.getMonth() + 1).padStart(2, "0");
  const dd = String(now.getDate()).padStart(2, "0");
  const todayStr = `${yyyy}-${mm}-${dd}`; // "2025-11-10"

  // tạo 24 ô từ 0h -> 23h
  const counts = Array(24).fill(0);

  trafficLogs.forEach((l) => {
    // backend mình đang ghi "ts": "2025-11-10 11:53:12"
    const rawTs = l.ts || l.created_at; // ưu tiên ts, nếu sau này bạn dùng supabase thì có created_at
    if (!rawTs) return;

    // tách "2025-11-10" và "11:53:12"
    const [datePart, timePart] = rawTs.split(" ");
    if (datePart !== todayStr) return; // không phải hôm nay thì bỏ

    const hourStr = timePart?.split(":")?.[0];
    const hour = Number(hourStr);
    if (!Number.isNaN(hour) && hour >= 0 && hour < 24) {
      counts[hour] = (counts[hour] || 0) + 1;
    }
  });

  const labels = Array.from({ length: 24 }, (_, h) =>
    (h < 10 ? "0" + h : h) + " giờ"
  );

  return { labels, counts };
})();


const trafficLineData = {
  labels: trafficToday.labels,
  datasets: [
    {
      label: "Lượt truy cập",
      data: trafficToday.counts,
      borderColor: "rgba(255, 159, 64, 1)",     // cam
      backgroundColor: "rgba(255, 159, 64, 0.15)", // nền mờ giống hình
      tension: 0.4,       // bo cong
      fill: true,         // tô dưới đường
      pointRadius: 4,
      pointBackgroundColor: "#000", // chấm đen
      borderWidth: 2,
      borderDash: [6, 4], // nét đứt giống hình
    },
  ],
};
const trafficLineOptions = {
  responsive: true,
  plugins: {
    legend: { display: false },
  },
  scales: {
    y: {
      beginAtZero: true,
      ticks: {
        stepSize: 1,
      },
    },
  },
};


  // ====== 2. chart top user dùng gợi ý ======
  const chartTopUsers = {
    labels: topSearchUsers.map((u) => u.full_name || u.user_id),
    datasets: [
      {
        label: "Số lần dùng gợi ý",
        data: topSearchUsers.map((u) => u.total_search || 0),
        backgroundColor: topSearchUsers.map(
          (_, i) => BAR_COLORS[i % BAR_COLORS.length]
        ),
        borderColor: topSearchUsers.map(
          (_, i) => BAR_COLORS[i % BAR_COLORS.length].replace("0.7", "1")
        ),
        borderWidth: 1,
      },
    ],
  };
  const topUsersOptions = {
    ...commonNoLegend,
    scales: {
      x: {
        ticks: {
          callback: function (value) {
            const label = this.getLabelForValue(value);
            if (!label) return "";
            const maxLen = 14;
            if (label.length <= maxLen) return label;
            return [label.slice(0, maxLen), label.slice(maxLen)];
          },
        },
      },
      y: { beginAtZero: true },
    },
  };

  // ====== 3. chart brand ======
  const brandChartData = {
    labels: brandFromLogs.map((b) => b.brand.toUpperCase()),
    datasets: [
      {
        label: "Số lần được tìm",
        data: brandFromLogs.map((b) => b.total),
        backgroundColor: brandFromLogs.map(
          (_, i) => BAR_COLORS[i % BAR_COLORS.length]
        ),
        borderColor: brandFromLogs.map(
          (_, i) => BAR_COLORS[i % BAR_COLORS.length].replace("0.7", "1")
        ),
        borderWidth: 1,
      },
    ],
  };

  // ====== 4. chart laptop click ======
  const clickedChartData = {
    labels: topClicked.map((x) => x.name || x.laptop_id),
    datasets: [
      {
        label: "Lượt click",
        data: topClicked.map((x) => x.total_click),
        backgroundColor: topClicked.map(
          (_, i) => BAR_COLORS[i % BAR_COLORS.length]
        ),
        borderColor: topClicked.map(
          (_, i) => BAR_COLORS[i % BAR_COLORS.length].replace("0.7", "1")
        ),
        borderWidth: 1,
      },
    ],
  };
  const clickedChartOptions = {
    ...commonNoLegend,
    scales: {
      x: {
        ticks: {
          callback: function (value) {
            const label = this.getLabelForValue(value);
            if (!label) return "";
            const maxLen = 12;
            if (label.length <= maxLen) return label;
            return [label.slice(0, maxLen), label.slice(maxLen, maxLen * 2)];
          },
        },
      },
      y: { beginAtZero: true },
    },
  };

  // ====== 5. chart laptop trong giỏ ======
  const cartChartData = {
    labels: topCart.map((x) => x.name || x.laptop_id),
    datasets: [
      {
        label: "Số lần nằm trong giỏ",
        data: topCart.map((x) => x.total_cart || 0),
        backgroundColor: topCart.map(
          (_, i) => BAR_COLORS[i % BAR_COLORS.length]
        ),
        borderColor: topCart.map(
          (_, i) => BAR_COLORS[i % BAR_COLORS.length].replace("0.7", "1")
        ),
        borderWidth: 1,
      },
    ],
  };
  const cartChartOptions = {
    ...clickedChartOptions,
  };

  return (
    <div>
      <h4 className="mb-3">📊 Thống kê</h4>
      {loading ? (
        <p>Đang tải Supabase...</p>
      ) : (
        <div className="row g-3 mb-4">
          <div className="col-md-3">
            <div className="card text-white bg-warning h-100">
              <div className="card-body d-flex justify-content-between align-items-center">
                <div>
                  <div className="fs-3 fw-bold">{stats.users}</div>
                  <div>Người dùng</div>
                </div>
                <i className="fas fa-users fa-3x opacity-50" />
              </div>
            </div>
          </div>

          <div className="col-md-3">
            <div className="card text-white bg-success h-100">
              <div className="card-body d-flex justify-content-between align-items-center">
                <div>
                  <div className="fs-3 fw-bold">{stats.laptops}</div>
                  <div>Laptop</div>
                </div>
                <i className="fas fa-laptop fa-3x opacity-50" />
              </div>
            </div>
          </div>

          <div className="col-md-3">
            <div className="card text-white bg-primary h-100">
              <div className="card-body d-flex justify-content-between align-items-center">
                <div>
                  <div className="fs-3 fw-bold">{stats.banners}</div>
                  <div>Banner</div>
                </div>
                <i className="fas fa-image fa-3x opacity-50" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* HÀNG 1: cột trái chia 2, cột phải vẫn như cũ */}
      <div className="row">
        {/* cột trái */}
        <div className="col-md-6 mb-4">
          {/* chart 1 */}
          <h5 className="mb-2">📈 Lượt gợi ý từ Flask (5 ngày gần nhất)</h5>
<div className="card p-3 h-40">
  {flaskLoading ? (
    <p>Đang tải...</p>
  ) : (
    <Bar data={flaskHorizontalData} options={flaskHorizontalOptions} />
  )}
</div>


          {/* chart 2: traffic */}
          <h6 className="mb-2">📶 Traffic theo giờ (hôm nay)</h6>
<div className="card p-3">
  {flaskLoading ? (
    <p>Đang tải...</p>
  ) : trafficToday.labels.length === 0 ? (
    <p>Hôm nay chưa có request.</p>
  ) : (
    <Line /* nếu muốn line, đổi sang <Line ... /> */ 
      data={trafficLineData}
      options={trafficLineOptions}
    />
  )}
</div>

        </div>

        {/* cột phải */}
        <div className="col-md-6 mb-4">
          <h5 className="mb-2">🏷️ Thương hiệu được tìm nhiều (7 hãng)</h5>
          <div className="card p-3 h-100">
            {flaskLoading ? (
              <p>Đang tải...</p>
            ) : (
              <>
                <Bar data={brandChartData} options={commonNoLegend} />

                <div className="table-responsive mt-3">
                  <table className="table table-sm table-striped">
                    <thead>
                      <tr>
                        <th>Thương hiệu</th>
                        <th className="text-end">Số lần tìm</th>
                      </tr>
                    </thead>
                    <tbody>
                      {brandFromLogs.map((row) => (
                        <tr key={row.brand}>
                          <td>{row.brand.toUpperCase()}</td>
                          <td className="text-end">{row.total}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* HÀNG 2: top user (1 mình 1 hàng) */}
      <div className="row">
        <div className="col-12 mb-4">
          <h5 className="mb-2">👤 Top người dùng gọi gợi ý nhiều nhất</h5>
          <div className="card p-3">
            {flaskLoading ? (
              <p>Đang tải...</p>
            ) : topSearchUsers.length === 0 ? (
              <p>Chưa có log user nào.</p>
            ) : (
              <>
                <Bar data={chartTopUsers} options={topUsersOptions} />
                <div className="d-flex flex-wrap gap-3 mt-3">
                  {topSearchUsers.map((u) => (
                    <div
                      key={u.user_id}
                      className="d-flex align-items-center gap-2"
                    >
                      <img
                        src={u.avatar_url || "https://placehold.co/40x40"}
                        alt={u.full_name}
                        style={{
                          width: 40,
                          height: 40,
                          borderRadius: "50%",
                          objectFit: "cover",
                        }}
                      />
                      <div>
                        <div>{u.full_name || u.user_id}</div>
                        <small className="text-muted">
                          {u.total_search} lần
                        </small>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* HÀNG 3: laptop click + laptop trong giỏ */}
      <div className="row">
        <div className="col-md-6 mb-4">
          <h5 className="mb-2">🖱️ Laptop được click nhiều nhất</h5>
          <div className="card p-3 h-100">
            {flaskLoading ? (
              <p>Đang tải...</p>
            ) : topClicked.length === 0 ? (
              <p>Chưa có click nào.</p>
            ) : (
              <>
                <Bar data={clickedChartData} options={clickedChartOptions} />

                <div className="table-responsive mt-3">
                  <table className="table table-sm table-hover align-middle">
                    <thead>
                      <tr>
                        <th>Laptop</th>
                        <th className="text-end">Số click</th>
                      </tr>
                    </thead>
                    <tbody>
                      {topClicked.map((row) => (
                        <tr key={row.laptop_id}>
                          <td className="d-flex align-items-center gap-2">
                            {row.image_url ? (
                              <img
                                src={row.image_url}
                                alt={row.name}
                                style={{
                                  width: 32,
                                  height: 32,
                                  objectFit: "cover",
                                  borderRadius: 6,
                                }}
                              />
                            ) : (
                              <div
                                style={{
                                  width: 32,
                                  height: 32,
                                  background: "#eee",
                                  borderRadius: 6,
                                }}
                              />
                            )}
                            <span>{row.name}</span>
                          </td>
                          <td className="text-end fw-bold">
                            {row.total_click}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="col-md-6 mb-4">
          <h5 className="mb-2">🛒 Laptop nằm trong giỏ nhiều nhất</h5>
          <div className="card p-3 h-100">
            {flaskLoading ? (
              <p>Đang tải...</p>
            ) : topCart.length === 0 ? (
              <p>
                Chưa có dữ liệu giỏ hàng (cho Flask trả thêm
                <code> top_cart_laptops </code>).
              </p>
            ) : (
              <Bar data={cartChartData} options={cartChartOptions} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
