// src/components/Topbar.jsx
import React, { useEffect, useState } from "react";
import { supabase } from "../supabaseClient";

export default function Topbar() {
  const [profile, setProfile] = useState({
    full_name: "",
    email: "",
    avatar_url: "",
  });

  useEffect(() => {
    const load = async () => {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (!session?.user) return;

      const { data, error } = await supabase
        .from("profiles")
        .select("full_name, email, avatar_url")
        .eq("id", session.user.id)
        .maybeSingle();

      if (!error && data) {
        setProfile({
          full_name: data.full_name || session.user.email,
          email: data.email || session.user.email,
          avatar_url:
            data.avatar_url ||
            `https://ui-avatars.com/api/?background=F97316&color=fff&name=${
              data.full_name || "Admin"
            }`,
        });
      } else {
        setProfile({
          full_name: session.user.email,
          email: session.user.email,
          avatar_url: `https://ui-avatars.com/api/?background=F97316&color=fff&name=${session.user.email}`,
        });
      }
    };

    load();
  }, []);

  return (
    <div
      className="d-flex justify-content-between align-items-center px-4 py-3 mb-3"
      style={{ background: "#fff", borderBottom: "1px solid #e5e7eb", boxShadow: "rgba(24, 24, 23, 0.64) -1px 1px 15px", }}
    >
      <div>
        {/* dòng này to + màu cam */}
        <div
          style={{
            fontSize: 20,
            fontWeight: 700,
            color: "#F97316",
            lineHeight: 1.2,
          }}
        >
          Chào mừng quản trị viên đã quay trở lại !
        </div>
        {/* tên admin */}
        
      </div>

      <div className="d-flex align-items-center gap-3">
        <div style={{ fontSize: 16, fontWeight: 600, color: "#111827" }}>
          {profile.full_name || "Admin"}
          <p><small style={{ opacity: 0.55 }}>{profile.email}</small></p>
        </div>
        
        <img
          src={
            profile.avatar_url ||
            "https://ui-avatars.com/api/?background=F97316&color=fff&name=A"
          }
          alt="avatar"
          width={46}
          height={46}
          style={{
            borderRadius: "50%",
            objectFit: "cover",
            border: "2px solid #F97316",
            boxShadow: "0 0 0 3px rgba(249,115,22,0.15)",
          }}
        />
      </div>
    </div>
  );
}
