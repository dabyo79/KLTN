// src/Login.jsx
import React from "react";
import { supabase } from "../supabaseClient";

export default function Login() {
  const handleLogin = async () => {
    await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: "http://localhost:5173", // giữ như hôm nãy
      },
    });
  };

  return (
    <div className="login-page">
      <div className="login-card">
        <h2 className="mb-3">Đăng nhập Admin</h2>
        <p className="mb-4">Đăng nhập bằng Google đã được cấp quyền</p>
        <button className="btn-google" onClick={handleLogin}>
            <i className="fa-brands fa-google me-2"></i>
            Đăng nhập với Google
        </button>
      </div>
    </div>
  );
}
