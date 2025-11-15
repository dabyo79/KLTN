import { useEffect, useState } from "react";
import { supabase } from "../supabaseClient";
import "bootstrap/dist/js/bootstrap.bundle.min.js";
const BUCKET = "laptopbanchon";
const BRAND_OPTIONS = ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI"];

export default function Products() {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({
    id: null,
    name: "",
    price: "",
    brand: "",
    description: "",
    image: "",
  });
  const [file, setFile] = useState(null);

  const load = async () => {
    const { data, error } = await supabase
      .from("products")
      .select("*")
      .order("created_at", { ascending: false });
    if (!error) setProducts(data ?? []);
  };

  useEffect(() => {
    load();
  }, []);

  const resetForm = () => {
    setForm({
      id: null,
      name: "",
      price: "",
      brand: "",
      description: "",
      image: "",
    });
    setFile(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    let imageUrl = form.image;

    // có file thì upload
    if (file) {
      const ext = file.name.split(".").pop();
      const filePath = `products/${crypto.randomUUID()}.${ext}`;

      const { error: uploadErr } = await supabase.storage
        .from(BUCKET)
        .upload(filePath, file, {
          upsert: true,
          contentType: file.type,
        });

      if (uploadErr) {
        alert("Upload ảnh thất bại: " + uploadErr.message);
        setLoading(false);
        return;
      }

      const { data: publicUrlData } = supabase.storage
        .from(BUCKET)
        .getPublicUrl(filePath);
      imageUrl = publicUrlData.publicUrl;
    }

    // UPDATE
    if (form.id) {
      const { error } = await supabase
        .from("products")
        .update({
          name: form.name,
          price: Number(form.price) || 0,
          brand: form.brand,
          description: form.description,
          image: imageUrl,
        })
        .eq("id", form.id);

      if (error) alert(error.message);
      else {
        await load();
        resetForm();
      }
    } else {
      // INSERT
      const { error } = await supabase.from("products").insert({
        name: form.name,
        price: Number(form.price) || 0,
        brand: form.brand,
        description: form.description,
        image: imageUrl,
      });

      if (error) alert(error.message);
      else {
        await load();
        resetForm();
      }
    }

    setLoading(false);
  };

  const handleEdit = (p) => {
    setForm({
      id: p.id,
      name: p.name || "",
      price: p.price || "",
      brand: p.brand || "",
      description: p.description || "",
      image: p.image || "",
    });
    setFile(null);
  };

  const handleDelete = async (p) => {
    const ok = confirm(`Xoá sản phẩm "${p.name}" ?`);
    if (!ok) return;
    const { error } = await supabase.from("products").delete().eq("id", p.id);
    if (error) alert(error.message);
    else load();
  };

  return (
    <div>
      <h4>📦 Sản phẩm</h4>

      <form className="row g-3 mb-4" onSubmit={handleSubmit}>
        <div className="col-md-4">
          <label className="form-label">Tên</label>
          <input
            className="form-control"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            required
          />
        </div>

        <div className="col-md-2">
          <label className="form-label">Giá</label>
          <input
            className="form-control"
            type="number"
            value={form.price}
            onChange={(e) => setForm({ ...form, price: e.target.value })}
            required
          />
        </div>

        {/* DROPDOWN BRAND */}
        <div className="col-md-3">
          <label className="form-label d-block">Thương hiệu</label>
          <div className="dropdown">
            <button
              className="btn btn-outline-secondary w-100 d-flex justify-content-between align-items-center"
              type="button"
              data-bs-toggle="dropdown"
              aria-expanded="false"
            >
              {form.brand ? form.brand : "— Chọn thương hiệu —"}
              <i className="fa fa-chevron-down ms-2 small"></i>
            </button>
            <ul className="dropdown-menu w-100">
              {BRAND_OPTIONS.map((b) => (
                <li key={b}>
                  <button
                    type="button"
                    className="dropdown-item"
                    onClick={() => setForm({ ...form, brand: b })}
                  >
                    {b}
                  </button>
                </li>
              ))}
              <li>
                <hr className="dropdown-divider" />
              </li>
              <li>
                <button
                    type="button"
                    className="dropdown-item text-danger"
                    onClick={() => setForm({ ...form, brand: "" })}
                  >
                    Xoá chọn
                  </button>
              </li>
            </ul>
          </div>
        </div>

        <div className="col-md-3">
          <label className="form-label">Ảnh (chọn file)</label>
          <input
            className="form-control"
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </div>

        <div className="col-md-12">
          <label className="form-label">Mô tả</label>
          <textarea
            className="form-control"
            rows="2"
            value={form.description}
            onChange={(e) =>
              setForm({ ...form, description: e.target.value })
            }
          />
        </div>

        <div className="col-12">
          <button className="btn btn-primary" disabled={loading}>
            {form.id ? "💾 Lưu thay đổi" : "+ Thêm mới"}
          </button>
          {form.id && (
            <button
              type="button"
              className="btn btn-secondary ms-2"
              onClick={resetForm}
            >
              Huỷ
            </button>
          )}
        </div>
      </form>

      <div className="table-responsive">
        <table className="table table-bordered align-middle">
          <thead>
            <tr>
              <th>Ảnh</th>
              <th>Tên</th>
              <th>Giá</th>
              <th>Thương hiệu</th>
              <th width="180">Hành động</th>
            </tr>
          </thead>
          <tbody>
            {products.map((p) => (
              <tr key={p.id}>
                <td>
                  {p.image ? (
                    <img
                      src={p.image}
                      alt=""
                      width="50"
                      height="50"
                      style={{ objectFit: "cover", borderRadius: 8 }}
                    />
                  ) : (
                    "—"
                  )}
                </td>
                <td>{p.name}</td>
                <td>{p.price?.toLocaleString("vi-VN")}</td>
                <td>{p.brand}</td>
                <td>
                  <button
                    className="btn btn-sm btn-outline-success me-2"
                    onClick={() => handleEdit(p)}
                  >
                    Sửa
                  </button>
                  <button
                    className="btn btn-sm btn-outline-danger"
                    onClick={() => handleDelete(p)}
                  >
                    Xoá
                  </button>
                </td>
              </tr>
            ))}
            {products.length === 0 && (
              <tr>
                <td colSpan="5" className="text-center text-muted">
                  Không có sản phẩm
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
