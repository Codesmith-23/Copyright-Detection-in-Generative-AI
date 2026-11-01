"use client";

import { useState, useEffect, FormEvent } from "react";
import { useRouter } from "next/navigation";
import {
  Shield,
  Upload,
  CheckCircle,
  AlertCircle,
  User,
  Mail,
  Camera,
  UserCheck,
} from "lucide-react";
import Button from "@/components/ui/Button";
import { authService } from "@/lib/auth";
import withAuth from "@/lib/withAuth";

const BACKEND_URL = "http://127.0.0.1:5000";

// Creator routes require the logged-in user's token.
const authHeaders = (): Record<string, string> => {
  const token = authService.getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
};

interface FormState {
  name: string;
  email: string;
  entityType: "creator" | "celebrity" | "brand_character";
  consent: boolean;
}

interface UploadState {
  entityId: string;
  files: FileList | null;
  uploading: boolean;
  uploadMessage: string;
}

function CreatorRegistrationPage() {
  const router = useRouter();

  // Step 1: Registration Form
  const [step, setStep] = useState<"register" | "upload" | "verify">(
    "register",
  );
  const [formData, setFormData] = useState<FormState>({
    name: "",
    email: "",
    entityType: "creator",
    consent: false,
  });
  const [regError, setRegError] = useState("");
  const [regLoading, setRegLoading] = useState(false);

  // Step 2: Photo Upload
  const [uploadData, setUploadData] = useState<UploadState>({
    entityId: "",
    files: null,
    uploading: false,
    uploadMessage: "",
  });

  // Step 3: Verification
  const [verificationStatus, setVerificationStatus] = useState<
    "pending" | "processing" | "verified" | "failed"
  >("pending");

  // Load logged-in user's data on mount
  useEffect(() => {
    const user = authService.getUser();
    if (user) {
      setFormData((prev) => ({
        ...prev,
        name: user.username || "",
        email: user.email || "",
      }));
    }
  }, []);

  // ============================================================================
  // STEP 1: REGISTRATION
  // ============================================================================

  const handleRegisterChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>,
  ) => {
    const { name, value, type } = e.target as HTMLInputElement;
    if (type === "checkbox") {
      setFormData({
        ...formData,
        [name]: (e.target as HTMLInputElement).checked,
      });
    } else {
      setFormData({ ...formData, [name]: value });
    }
  };

  const validateRegistration = (): string | null => {
    if (!formData.name.trim()) return "Name is required";
    if (!formData.email.includes("@")) return "Valid email required";
    if (!formData.consent) return "You must consent to biometric data storage";
    return null;
  };

  const handleRegisterSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setRegError("");
    setRegLoading(true);

    const err = validateRegistration();
    if (err) {
      setRegError(err);
      setRegLoading(false);
      return;
    }

    try {
      const response = await fetch(`${BACKEND_URL}/creators/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({
          name: formData.name,
          email: formData.email,
          type: formData.entityType,
          consent: formData.consent,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || "Registration failed");
      }

      const data = await response.json();
      setUploadData({
        entityId: data.entity_id,
        files: null,
        uploading: false,
        uploadMessage: "",
      });
      setStep("upload");
    } catch (error: any) {
      setRegError(error.message || "Registration failed");
    } finally {
      setRegLoading(false);
    }
  };

  // ============================================================================
  // STEP 2: PHOTO UPLOAD
  // ============================================================================

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUploadData({ ...uploadData, files: e.target.files });
  };

  const handlePhotoUpload = async (e: FormEvent) => {
    e.preventDefault();

    if (!uploadData.files || uploadData.files.length === 0) {
      setUploadData({
        ...uploadData,
        uploadMessage: "Please select at least one image",
      });
      return;
    }

    setUploadData({ ...uploadData, uploading: true, uploadMessage: "" });

    try {
      const formDataToSend = new FormData(); // Renamed to avoid conflict with state 'formData'
      formDataToSend.append("entity_id", uploadData.entityId);

      for (let i = 0; i < uploadData.files.length; i++) {
        formDataToSend.append("files", uploadData.files[i]);
      }

      const response = await fetch(`${BACKEND_URL}/creators/upload-reference`, {
        method: "POST",
        headers: authHeaders(),
        body: formDataToSend,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || "Upload failed");
      }

      const data = await response.json();
      setUploadData({
        ...uploadData,
        uploading: false,
        uploadMessage: `Success! ${data.embeddings_stored} embeddings stored.`,
      });

      // Move to verification step
      setTimeout(() => setStep("verify"), 1500);
    } catch (error: any) {
      setUploadData({
        ...uploadData,
        uploading: false,
        uploadMessage: error.message || "Upload failed",
      });
    }
  };

  // ============================================================================
  // STEP 3: VERIFICATION
  // ============================================================================

  // Checks with the backend that reference embeddings were stored and the
  // identity is active for matching.
  const handleStartVerification = async () => {
    setVerificationStatus("processing");
    try {
      const response = await fetch(
        `${BACKEND_URL}/creators/profile/${encodeURIComponent(uploadData.entityId)}`,
        { headers: authHeaders() },
      );
      const data = response.ok ? await response.json() : null;
      if (data?.protection_status === "active") {
        setVerificationStatus("verified");
        // Redirect to detect page after successful verification
        setTimeout(() => router.push("/detect"), 3000);
      } else {
        setVerificationStatus("failed");
      }
    } catch {
      setVerificationStatus("failed");
    }
  };

  // ============================================================================
  // RENDER
  // ============================================================================

  if (step === "register") {
    return (
      <div className="w-full bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 py-12">
        <div className="container mx-auto px-4">
          <div className="max-w-md mx-auto">
            {/* Card */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden">
              {/* Header */}
              <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-8">
                <div className="flex items-center gap-3 mb-2">
                  <Shield className="w-8 h-8 text-white" />
                  <h1 className="text-2xl font-bold text-white">
                    Register Your Identity
                  </h1>
                </div>
                <p className="text-blue-100 text-sm">
                  Protect your likeness from unauthorized AI use
                </p>
              </div>

              {/* Form */}
              <form onSubmit={handleRegisterSubmit} className="p-6 space-y-4">
                {/* Name */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    <User className="w-4 h-4 inline mr-2" />
                    Full Name
                  </label>
                  <input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleRegisterChange}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                    placeholder="e.g., Sarah Connor"
                  />
                </div>

                {/* Email */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    <Mail className="w-4 h-4 inline mr-2" />
                    Email Address
                  </label>
                  <input
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={handleRegisterChange}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                    placeholder="sarah@example.com"
                  />
                </div>

                {/* Entity Type */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    <UserCheck className="w-4 h-4 inline mr-2" />
                    Identity Type
                  </label>
                  <select
                    name="entityType"
                    value={formData.entityType}
                    onChange={handleRegisterChange}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  >
                    <option value="creator">Creator / Influencer</option>
                    <option value="celebrity">Celebrity / Public Figure</option>
                    <option value="brand_character">
                      Brand Character / Mascot
                    </option>
                  </select>
                </div>

                {/* Consent */}
                <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                  <label className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      name="consent"
                      checked={formData.consent}
                      onChange={handleRegisterChange}
                      className="mt-1 w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      I consent to storing my facial biometric data to detect
                      unauthorized AI-generated content using my likeness.
                    </span>
                  </label>
                </div>

                {/* Error */}
                {regError && (
                  <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3 flex items-start gap-2">
                    <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                    <p className="text-sm text-red-700 dark:text-red-300">
                      {regError}
                    </p>
                  </div>
                )}

                {/* Submit */}
                <Button
                  type="submit"
                  disabled={regLoading}
                  className="w-full"
                  size="lg"
                >
                  {regLoading ? "Registering..." : "Next: Upload References"}
                </Button>
              </form>

              {/* Footer */}
              <div className="border-t border-gray-200 dark:border-gray-700 px-6 py-4 text-center text-sm text-gray-600 dark:text-gray-400">
                Already registered?{" "}
                <a
                  href="/detect"
                  className="text-blue-600 hover:underline font-semibold"
                >
                  Go to Detection
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ============================================================================
  // STEP 2: UPLOAD PHOTOS
  // ============================================================================

  if (step === "upload") {
    return (
      <div className="w-full bg-gradient-to-br from-green-50 to-emerald-100 dark:from-gray-900 dark:to-gray-800 py-12">
        <div className="container mx-auto px-4">
          <div className="max-w-md mx-auto">
            {/* Card */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden">
              {/* Header */}
              <div className="bg-gradient-to-r from-green-600 to-emerald-600 px-6 py-8">
                <div className="flex items-center gap-3 mb-2">
                  <Camera className="w-8 h-8 text-white" />
                  <h1 className="text-2xl font-bold text-white">
                    Upload Reference Photos
                  </h1>
                </div>
                <p className="text-green-100 text-sm">
                  ID: {uploadData.entityId}
                </p>
              </div>

              {/* Form */}
              <form onSubmit={handlePhotoUpload} className="p-6 space-y-4">
                {/* File Input */}
                <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-6 text-center hover:border-green-500 transition">
                  <Upload className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                  <label className="block">
                    <span className="text-blue-600 hover:text-blue-700 font-semibold cursor-pointer">
                      Click to upload
                    </span>{" "}
                    or drag and drop
                    <input
                      type="file"
                      multiple
                      accept="image/*"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                  </label>
                  <p className="text-xs text-gray-500 mt-2">
                    PNG, JPG, GIF up to 10MB (multiple images recommended)
                  </p>
                  {uploadData.files && (
                    <p className="text-sm text-green-600 font-semibold mt-2">
                      ✓ {uploadData.files.length} file(s) selected
                    </p>
                  )}
                </div>

                {/* Info Box */}
                <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 text-sm text-blue-700 dark:text-blue-300">
                  <p className="font-semibold mb-1">
                    📸 Tips for best results:
                  </p>
                  <ul className="space-y-1 text-xs">
                    <li>• Clear, front-facing photos</li>
                    <li>• Good lighting, no heavy shadows</li>
                    <li>• Multiple angles improve accuracy</li>
                    <li>• At least 5-10 different photos recommended</li>
                  </ul>
                </div>

                {/* Message */}
                {uploadData.uploadMessage && (
                  <div
                    className={`p-3 rounded-lg flex items-start gap-2 ${
                      uploadData.uploadMessage.includes("Success")
                        ? "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800"
                        : "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800"
                    }`}
                  >
                    {uploadData.uploadMessage.includes("Success") ? (
                      <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
                    ) : (
                      <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                    )}
                    <p
                      className={`text-sm ${
                        uploadData.uploadMessage.includes("Success")
                          ? "text-green-700 dark:text-green-300"
                          : "text-red-700 dark:text-red-300"
                      }`}
                    >
                      {uploadData.uploadMessage}
                    </p>
                  </div>
                )}

                {/* Submit */}
                <Button
                  type="submit"
                  disabled={uploadData.uploading || !uploadData.files}
                  className="w-full"
                  size="lg"
                >
                  {uploadData.uploading ? "Uploading..." : "Next: Verification"}
                </Button>

                {/* Skip Button */}
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => router.push("/detect")}
                  className="w-full"
                >
                  Skip for Now
                </Button>
              </form>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ============================================================================
  // STEP 3: VERIFICATION
  // ============================================================================

  return (
    <div className="w-full bg-gradient-to-br from-purple-50 to-pink-100 dark:from-gray-900 dark:to-gray-800 py-12">
      <div className="container mx-auto px-4">
        <div className="max-w-md mx-auto">
          {/* Card */}
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden">
            {/* Header */}
            <div className="bg-gradient-to-r from-purple-600 to-pink-600 px-6 py-8">
              <div className="flex items-center gap-3 mb-2">
                <Shield className="w-8 h-8 text-white" />
                <h1 className="text-2xl font-bold text-white">
                  Identity Verification
                </h1>
              </div>
              <p className="text-purple-100 text-sm">
                ID: {uploadData.entityId}
              </p>
            </div>

            {/* Verification Status */}
            <div className="p-6 space-y-6">
              {/* Pending State */}
              {verificationStatus === "pending" && (
                <div className="text-center space-y-4">
                  <div className="w-20 h-20 mx-auto bg-purple-100 dark:bg-purple-900/20 rounded-full flex items-center justify-center">
                    <UserCheck className="w-10 h-10 text-purple-600 dark:text-purple-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                      Ready for Verification
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400 text-sm">
                      Start the verification process to protect your identity.
                      This ensures your biometric data is securely stored and
                      validated.
                    </p>
                  </div>
                  <Button
                    onClick={handleStartVerification}
                    size="lg"
                    className="w-full"
                  >
                    Start Verification
                  </Button>
                </div>
              )}

              {/* Processing State */}
              {verificationStatus === "processing" && (
                <div className="text-center space-y-4">
                  <div className="w-20 h-20 mx-auto bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center animate-pulse">
                    <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                      Verifying Your Identity...
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400 text-sm">
                      We're analyzing your images and validating biometric
                      patterns. This may take a few moments.
                    </p>
                  </div>
                  <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                    <ul className="space-y-2 text-sm text-blue-700 dark:text-blue-300 text-left">
                      <li className="flex items-center gap-2">
                        <CheckCircle className="w-4 h-4" />
                        Analyzing facial features
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle className="w-4 h-4" />
                        Extracting biometric embeddings
                      </li>
                      <li className="flex items-center gap-2 opacity-50">
                        <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                        Validating authenticity
                      </li>
                    </ul>
                  </div>
                </div>
              )}

              {/* Verified State */}
              {verificationStatus === "verified" && (
                <div className="text-center space-y-4">
                  <div className="w-20 h-20 mx-auto bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center">
                    <CheckCircle className="w-10 h-10 text-green-600 dark:text-green-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                      ✓ Verification Complete!
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400 text-sm">
                      Your identity has been successfully verified and is now
                      protected. Redirecting you to the detection page...
                    </p>
                  </div>
                  <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4 text-sm text-green-700 dark:text-green-300">
                    <p className="font-semibold mb-1">What's Next?</p>
                    <ul className="space-y-1 text-xs text-left">
                      <li>
                        • Your likeness is now tracked across AI-generated
                        content
                      </li>
                      <li>• Unauthorized use will be flagged automatically</li>
                      <li>• You'll receive alerts for potential violations</li>
                    </ul>
                  </div>
                  <Button
                    onClick={() => router.push("/detect")}
                    size="lg"
                    className="w-full"
                  >
                    Go to Detection Page
                  </Button>
                </div>
              )}

              {/* Failed State */}
              {verificationStatus === "failed" && (
                <div className="text-center space-y-4">
                  <div className="w-20 h-20 mx-auto bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center">
                    <AlertCircle className="w-10 h-10 text-red-600 dark:text-red-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                      Verification Failed
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400 text-sm">
                      We couldn't verify your identity with the provided images.
                      Please try again with clearer, front-facing photos.
                    </p>
                  </div>
                  <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 text-sm text-red-700 dark:text-red-300 text-left">
                    <p className="font-semibold mb-1">Common Issues:</p>
                    <ul className="space-y-1 text-xs">
                      <li>• Photos too dark or blurry</li>
                      <li>• Face partially obscured</li>
                      <li>• Inconsistent lighting</li>
                      <li>• Too few reference images</li>
                    </ul>
                  </div>
                  <div className="flex gap-3">
                    <Button
                      variant="outline"
                      onClick={() => setStep("upload")}
                      className="flex-1"
                    >
                      Upload New Photos
                    </Button>
                    <Button
                      onClick={handleStartVerification}
                      className="flex-1"
                    >
                      Retry Verification
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default withAuth(CreatorRegistrationPage);
