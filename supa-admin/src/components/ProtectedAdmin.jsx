// src/components/ProtectedAdmin.jsx
import React, { useEffect, useState } from "react";
import { supabase } from "../supabaseClient";

export default function ProtectedAdmin({ children }) {
  const [loading, setLoading] = useState(true);
  const [session, setSession] = useState(null);
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    const check = async () => {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      setSession(session);

      if (session?.user) {
        // lấy profile
        const { data: profiles } = await supabase
          .from("profiles")
          .select("role")
          .eq("id", session.user.id)
          .maybeSingle();

        setIsAdmin(profiles?.role === "admin");
      }

      setLoading(false);
    };

    check();

    const { data: sub } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });

    return () => {
      sub.subscription.unsubscribe();
    };
  }, []);

  const handleLogin = async () => {
    await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: window.location.origin, // quay lại trang này
      },
    });
  };

  if (loading) {
    return <div className="login-wrapper">Đang kiểm tra…</div>;
  }

  // chưa login
  if (!session) {
    return (
      <div className="login-wrapper">
        <div className="login-box">
          <h2>Đăng nhập admin</h2>
          <p className="login-sub">Vui lòng dùng Gmail đã được cấp quyền</p>
          <button className="btn-google" onClick={handleLogin}>
            <i className="fa-brands fa-google me-2"></i> Đăng nhập bằng Google
          </button>
        </div>
      </div>
    );
  }

  // login rồi nhưng không phải admin
  if (session && !isAdmin) {
    return (
      <div className="login-wrapper">
        <div className="login-box">
          <h2>Tài khoản không có quyền</h2>
          <p className="login-sub">Liên hệ để được cấp quyền admin.</p>
          <button
            className="btn-google"
            onClick={() => supabase.auth.signOut()}
          >
            Đăng xuất
          </button>
        </div>
      </div>
    );
  }

  // ok, là admin
  return children;
}
