import axios, { AxiosProgressEvent } from "axios";
import {
  DetectionApiResponse,
  HistoryEntry,
  ApiError,
  AuthResponse,
  LoginCredentials,
  RegisterCredentials,
  VerifyResponse,
  SupportFormData,
} from "@/types";
import { authService } from "@/lib/auth";

// Force IPv4 to prevent 'ECONNREFUSED' errors on Windows
const API_BASE_URL = "http://127.0.0.1:5000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000, // 5 minutes (for large video uploads)
  headers: {
    "Content-Type": "application/json",
  },
});

// Request Interceptor (Auth)
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("auth_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error),
);

// Login and register answer 401 for a wrong password. That is not an expired
// session, so those calls keep their own error message and never redirect.
const AUTH_FORM_PATHS = ["/auth/login", "/auth/register"];

// Response Interceptor (Auth Error Handling)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const url: string = error.config?.url || "";
    const isAuthForm = AUTH_FORM_PATHS.some((path) => url.startsWith(path));
    if (error.response?.status === 401 && !isAuthForm) {
      // Session expired or invalid: clear token and user, then go to login.
      authService.logout();
      if (window.location.pathname !== "/login") {
        window.location.href = "/login";
      }
    }
    return Promise.reject(error);
  },
);

// Turns an API failure into a message for the user. The backend sends
// {"error": "text"}; older code expected {"error": {"message": "text"}}.
export function errorMessage(error: any, fallback: string): string {
  if (axios.isAxiosError(error) && !error.response) {
    return error.code === "ECONNABORTED"
      ? "The server took too long to respond. Please try again."
      : "Cannot reach the server. Check that the backend is running.";
  }
  const apiError = error?.response?.data?.error as string | ApiError | undefined;
  if (typeof apiError === "string" && apiError) return apiError;
  if (apiError && typeof apiError === "object" && apiError.message) {
    return apiError.message;
  }
  return fallback;
}

export const apiService = {
  // --- ANALYZE MEDIA (Video or Audio) ---
  async analyzeVideo(
    file: File,
    onUploadProgress?: (progress: number) => void,
    onProcessingStage?: (stage: string, progress: number) => void,
  ): Promise<DetectionApiResponse> {
    const formData = new FormData();
    formData.append("file", file);

    try {
      // ✅ FIX 1: Removed 'ApiResponse' wrapper.
      // We expect raw 'DetectionApiResponse' from the backend.
      const response = await api.post<DetectionApiResponse>(
        "/predict/media",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          onUploadProgress: (progressEvent: AxiosProgressEvent) => {
            if (progressEvent.total && onUploadProgress) {
              const progress = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total,
              );
              onUploadProgress(progress);
            }
          },
        },
      );

      // UI Simulation (Optional visual feedback)
      if (onProcessingStage) {
        onProcessingStage("Analyzing frames...", 40);
        await new Promise((r) => setTimeout(r, 400));
        onProcessingStage("Detecting anomalies...", 70);
        await new Promise((r) => setTimeout(r, 400));
        onProcessingStage("Finalizing results...", 90);
        await new Promise((r) => setTimeout(r, 400));
      }

      // ✅ FIX 2: Return data directly
      return response.data;
    } catch (error: any) {
      throw new Error(errorMessage(error, "Analysis failed"));
    }
  },

  // --- GET HISTORY ---
  async getHistory(): Promise<HistoryEntry[]> {
    try {
      // ✅ FIX 3: Expect raw array, not wrapped
      const response = await api.post<HistoryEntry[]>("/history", {});
      return response.data;
    } catch (error: any) {
      throw new Error(errorMessage(error, "Failed to fetch history"));
    }
  },

  // --- DELETE HISTORY ---
  // Accepts string or number to be safe
  async deleteHistoryEntry(resultId: string | number): Promise<void> {
    try {
      await api.delete(`/history/delete/${resultId}`);
    } catch (error: any) {
      throw new Error(errorMessage(error, "Failed to delete history entry"));
    }
  },

  // --- REPORT RESULT (Optional Stub) ---
  async reportResult(
    resultId: string,
    type: "false_positive" | "false_negative",
  ): Promise<void> {
    console.log(`Reporting ${type} for ${resultId}`);
    return Promise.resolve();
  },

  // --- AUTHENTICATION ---
  async register(credentials: RegisterCredentials): Promise<AuthResponse> {
    try {
      const response = await api.post<AuthResponse>(
        "/auth/register",
        credentials,
      );
      return response.data;
    } catch (error: any) {
      throw new Error(errorMessage(error, "Registration failed"));
    }
  },

  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    try {
      const response = await api.post<AuthResponse>("/auth/login", credentials);
      return response.data;
    } catch (error: any) {
      throw new Error(errorMessage(error, "Login failed"));
    }
  },

async submitSupportForm(formData: SupportFormData): Promise<void> {
  try {
    await api.post('/support', formData);
  } catch (error: any) {
    // Silently succeed even if backend has no /support endpoint
    console.warn('Support form submission:', error.message);
  }
},

  async verify(): Promise<VerifyResponse> {
    try {
      const response = await api.get<VerifyResponse>("/auth/verify");
      return response.data;
    } catch (error: any) {
      throw new Error(errorMessage(error, "Token verification failed"));
    }
  },
};

export const detectionAPI = apiService;
