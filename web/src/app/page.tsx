"use client";
import { Footer } from "@/app/components/footer";
// import { Logo } from "@/app/components/logo";
import { PresetQuery } from "@/app/components/preset-query";
import { Search } from "@/app/components/search";
import React from "react";

export default function Home() {
  return (
    <div className="absolute inset-0 min-h-[500px] flex items-center justify-center">
      <div className="relative flex flex-col gap-8 px-4 -mt-24">
        <Search></Search>
        <div className="flex gap-2 flex-wrap justify-center">
          <PresetQuery query="手机无法获得IP怎么办？"></PresetQuery>
          <PresetQuery query="手机信号正常，但使用移动数据无法上网怎么办？"></PresetQuery>
        </div>
        <Footer></Footer>
      </div>
    </div>
  );
}
