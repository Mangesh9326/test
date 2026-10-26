import { NextResponse } from "next/server";

export async function GET() {
  try {
    const apiKey = process.env.WINDY_API_KEY;

    if (!apiKey) {
      return NextResponse.json(
        { error: "WINDY_API_KEY is not configured" },
        { status: 500 }
      );
    }

    const url =
      "https://api.windy.com/webcams/api/v3/webcams" +
      "?nearby=19.0760,72.8777,50" +
      "&limit=100" +
      "&include=images,location,player,urls" +
      "&lang=en";

    const response = await fetch(url, {
      headers: {
        "x-windy-api-key": apiKey,
      },
      cache: "no-store",
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, {
        status: response.status,
      });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Windy API error:", error);

    return NextResponse.json(
      { error: "Failed to fetch Windy webcams" },
      { status: 500 }
    );
  }
}