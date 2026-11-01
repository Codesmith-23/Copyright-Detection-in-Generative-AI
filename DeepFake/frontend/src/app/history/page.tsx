"use client";

import React, { useEffect, useState } from "react";
import {
  History,
  Search,
  Filter,
  Trash2,
  Download,
  Eye,
  AlertTriangle,
  CheckCircle,
  Calendar,
  FileVideo,
  SortAsc,
  SortDesc,
} from "lucide-react";
import Link from "next/link";
import { useHistory } from "@/hooks/useHistory";
import { HistoryEntry } from "@/types";
import {
  cn,
  formatDate,
  formatFileSize,
  getConfidenceColor,
} from "@/lib/utils";
import Button from "@/components/ui/Button";
import Modal from "@/components/ui/Modal";
import withAuth from "@/lib/withAuth";

function HistoryPage() {
  const { isLoading, getHistoryfromAPI, removeEntry } = useHistory();
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState<"all" | "deepfake" | "authentic">(
    "all",
  );
  const [sortBy, setSortBy] = useState<"date" | "filename" | "confidence">(
    "date",
  );
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [selectedEntry, setSelectedEntry] = useState<HistoryEntry | null>(null);

  const [history, setHistory] = useState<HistoryEntry[]>([]);
  useEffect(() => {
    const data = getHistoryfromAPI();
    data.then((res) => {
      if (res) {
        setHistory(res);
      }
    });
  }, [getHistoryfromAPI]);

  const handleRemoveEntry = async (id: string) => {
    try {
      await removeEntry(id);
      setHistory((prev) => prev.filter((entry) => entry.id !== id));
      setSelectedEntry(null);
    } catch (error) {
      console.error("Failed to remove entry:", error);
    }
  };

  // Filter and sort history
  const filteredHistory = history
    .filter((entry) => {
      const matchesSearch = entry.file_name
        .toLowerCase()
        .includes(searchTerm.toLowerCase());
      const matchesFilter = filterType === "all" || entry.result === filterType;
      return matchesSearch && matchesFilter;
    })
    .sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case "date":
          comparison =
            new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
          break;
        case "filename":
          comparison = a.file_name.localeCompare(b.file_name);
          break;
        case "confidence":
          comparison = a.confidence - b.confidence;
          break;
      }

      return sortOrder === "asc" ? comparison : -comparison;
    });

  const handleSort = (newSortBy: "date" | "filename" | "confidence") => {
    if (sortBy === newSortBy) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortBy(newSortBy);
      setSortOrder("desc");
    }
  };

  const handleExportHistory = () => {
    const exportData = {
      exportDate: new Date().toISOString(),
      totalEntries: history.length,
      entries: history.map((entry) => ({
        ...entry,
        timestamp: entry.timestamp,
      })),
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataUri =
      "data:application/json;charset=utf-8," + encodeURIComponent(dataStr);

    const exportFileDefaultName = `deepfake-detection-history-${new Date().toISOString().split("T")[0]}.json`;

    const linkElement = document.createElement("a");
    linkElement.setAttribute("href", dataUri);
    linkElement.setAttribute("download", exportFileDefaultName);
    linkElement.click();
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading history...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center space-x-3 mb-4">
            <History className="h-8 w-8 text-primary-600" />
            <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white">
              Analysis History
            </h1>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl">
            View and manage your past video analysis results.
          </p>
        </div>

        {history.length === 0 ? (
          /* Empty State */
          <div className="text-center py-16">
            <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-12 max-w-md mx-auto">
              <FileVideo className="h-16 w-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                No Analysis History
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                You have not analyzed any videos yet. Start by uploading a video
                to see your results here.
              </p>
              <Link href="/detect">
                <Button>Start Analysis</Button>
              </Link>
            </div>
          </div>
        ) : (
          <>
            {/* Controls */}
            <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 mb-8">
              <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                {/* Search and Filter */}
                <div className="flex flex-col sm:flex-row gap-4 flex-1">
                  {/* Search */}
                  <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search by filename..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10 pr-4 py-2 w-full border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:text-white"
                    />
                  </div>

                  {/* Filter */}
                  <div className="relative">
                    <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                    <select
                      value={filterType}
                      onChange={(e) =>
                        setFilterType(
                          e.target.value as "all" | "deepfake" | "authentic",
                        )
                      }
                      className="pl-10 pr-8 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:text-white appearance-none"
                    >
                      <option value="all">All Results</option>
                      <option value="deepfake">Deepfakes</option>
                      <option value="authentic">Real Videos</option>
                    </select>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-3">
                  <Button
                    variant="outline"
                    onClick={handleExportHistory}
                    size="sm"
                  >
                    <Download size={16} />
                    Export
                  </Button>
                </div>
              </div>

              {/* Stats */}
              <div className="mt-6 flex flex-wrap gap-6 text-sm text-gray-600 dark:text-gray-400">
                <span>Total: {history.length} analyses</span>
                <span>Showing: {filteredHistory.length} results</span>
                <span>
                  Deepfakes:{" "}
                  {history.filter((e) => e.result === "deepfake").length}
                </span>
                <span>
                  Real: {history.filter((e) => e.result === "authentic").length}
                </span>
              </div>
            </div>

            {/* Results Table */}
            {filteredHistory.length === 0 ? (
              <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-8 text-center">
                <p className="text-gray-600 dark:text-gray-400">
                  No results found for your search criteria.
                </p>
              </div>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
                {/* Desktop Table */}
                <div className="hidden md:block overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-50 dark:bg-gray-700">
                      <tr>
                        <th className="px-6 py-4 text-left">
                          <button
                            onClick={() => handleSort("filename")}
                            className="flex items-center space-x-1 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
                          >
                            <span>File Name</span>
                            {sortBy === "filename" &&
                              (sortOrder === "asc" ? (
                                <SortAsc size={16} />
                              ) : (
                                <SortDesc size={16} />
                              ))}
                          </button>
                        </th>
                        <th className="px-6 py-4 text-left">
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Result
                          </span>
                        </th>
                        <th className="px-6 py-4 text-left">
                          <button
                            onClick={() => handleSort("confidence")}
                            className="flex items-center space-x-1 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
                          >
                            <span>Confidence</span>
                            {sortBy === "confidence" &&
                              (sortOrder === "asc" ? (
                                <SortAsc size={16} />
                              ) : (
                                <SortDesc size={16} />
                              ))}
                          </button>
                        </th>
                        <th className="px-6 py-4 text-left">
                          <button
                            onClick={() => handleSort("date")}
                            className="flex items-center space-x-1 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
                          >
                            <span>Date</span>
                            {sortBy === "date" &&
                              (sortOrder === "asc" ? (
                                <SortAsc size={16} />
                              ) : (
                                <SortDesc size={16} />
                              ))}
                          </button>
                        </th>
                        <th className="px-6 py-4 text-left">
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Size
                          </span>
                        </th>
                        <th className="px-6 py-4 text-right">
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Actions
                          </span>
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                      {filteredHistory.map((entry) => (
                        <tr
                          key={entry.id}
                          className="hover:bg-gray-50 dark:hover:bg-gray-700"
                        >
                          <td className="px-6 py-4">
                            <div className="flex items-center space-x-3">
                              <FileVideo className="h-5 w-5 text-gray-400" />
                              <span className="text-sm font-medium text-gray-900 dark:text-white truncate max-w-xs">
                                {entry.file_name}
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center space-x-2">
                              {entry.result === "deepfake" ? (
                                <AlertTriangle className="h-4 w-4 text-red-600" />
                              ) : (
                                <CheckCircle className="h-4 w-4 text-green-600" />
                              )}
                              <span
                                className={cn(
                                  "text-sm font-medium capitalize",
                                  entry.result === "deepfake"
                                    ? "text-red-600"
                                    : "text-green-600",
                                )}
                              >
                                {entry.result}
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center space-x-2">
                              <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2 max-w-20">
                                <div
                                  className={cn(
                                    "h-2 rounded-full",
                                    entry.result === "deepfake"
                                      ? "bg-red-500"
                                      : "bg-green-500",
                                  )}
                                  style={{ width: `${entry.confidence}%` }}
                                />
                              </div>
                              <span
                                className={cn(
                                  "text-sm font-medium",
                                  getConfidenceColor(entry.confidence),
                                )}
                              >
                                {entry.confidence}%
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center space-x-2">
                              <Calendar className="h-4 w-4 text-gray-400" />
                              <span className="text-sm text-gray-600 dark:text-gray-400">
                                {formatDate(entry.timestamp)}
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <span className="text-sm text-gray-600 dark:text-gray-400">
                              {formatFileSize(Number(entry.file_size))}
                            </span>
                          </td>
                          <td className="px-6 py-4 text-right">
                            <div className="flex items-center justify-end space-x-2">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setSelectedEntry(entry)}
                                className="text-primary-600 hover:text-primary-700"
                              >
                                <Eye size={16} />
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleRemoveEntry(entry.id)}
                                className="text-red-600 hover:text-red-700"
                              >
                                <Trash2 size={16} />
                              </Button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Mobile Cards */}
                <div className="md:hidden divide-y divide-gray-200 dark:divide-gray-700">
                  {filteredHistory.map((entry) => (
                    <div key={entry.id} className="p-6">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center space-x-2 flex-1 min-w-0">
                          <FileVideo className="h-5 w-5 text-gray-400 flex-shrink-0" />
                          <span className="text-sm font-medium text-gray-900 dark:text-white truncate">
                            {entry.file_name}
                          </span>
                        </div>
                        <div className="flex items-center space-x-2 ml-3">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setSelectedEntry(entry)}
                            className="text-primary-600 hover:text-primary-700"
                          >
                            <Eye size={16} />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleRemoveEntry(entry.id)}
                            className="text-red-600 hover:text-red-700"
                          >
                            <Trash2 size={16} />
                          </Button>
                        </div>
                      </div>

                      <div className="flex items-center space-x-4 mb-3">
                        <div className="flex items-center space-x-2">
                          {entry.result === "deepfake" ? (
                            <AlertTriangle className="h-4 w-4 text-red-600" />
                          ) : (
                            <CheckCircle className="h-4 w-4 text-green-600" />
                          )}
                          <span
                            className={cn(
                              "text-sm font-medium capitalize",
                              entry.result === "deepfake"
                                ? "text-red-600"
                                : "text-green-600",
                            )}
                          >
                            {entry.result}
                          </span>
                        </div>
                        <span
                          className={cn(
                            "text-sm font-medium",
                            getConfidenceColor(entry.confidence),
                          )}
                        >
                          {entry.confidence}% confidence
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
                        <span>{formatDate(entry.timestamp)}</span>
                        <span>{formatFileSize(Number(entry.file_size))}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* Entry Detail Modal */}
        {selectedEntry && (
          <Modal
            isOpen={!!selectedEntry}
            onClose={() => setSelectedEntry(null)}
            title="Analysis Details"
            size="lg"
          >
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    File Name
                  </h4>
                  <p className="text-lg font-medium text-gray-900 dark:text-white break-all">
                    {selectedEntry.file_name}
                  </p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    File Size
                  </h4>
                  <p className="text-lg font-medium text-gray-900 dark:text-white">
                    {formatFileSize(Number(selectedEntry.file_size))}
                  </p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Result
                  </h4>
                  <div className="flex items-center space-x-2">
                    {selectedEntry.result === "deepfake" ? (
                      <AlertTriangle className="h-5 w-5 text-red-600" />
                    ) : (
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    )}
                    <span
                      className={cn(
                        "text-lg font-medium capitalize",
                        selectedEntry.result === "deepfake"
                          ? "text-red-600"
                          : "text-green-600",
                      )}
                    >
                      {selectedEntry.result === "deepfake"
                        ? "Likely Deepfake"
                        : "Likely Real"}
                    </span>
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Confidence
                  </h4>
                  <div className="flex items-center space-x-3">
                    <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                      <div
                        className={cn(
                          "h-3 rounded-full",
                          selectedEntry.result === "deepfake"
                            ? "bg-red-500"
                            : "bg-green-500",
                        )}
                        style={{ width: `${selectedEntry.confidence}%` }}
                      />
                    </div>
                    <span
                      className={cn(
                        "text-lg font-medium",
                        getConfidenceColor(selectedEntry.confidence),
                      )}
                    >
                      {selectedEntry.confidence}%
                    </span>
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Analysis Date
                  </h4>
                  <p className="text-lg font-medium text-gray-900 dark:text-white">
                    {formatDate(selectedEntry.timestamp)}
                  </p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Analysis ID
                  </h4>
                  <p className="text-sm font-mono text-gray-900 dark:text-white break-all">
                    {selectedEntry.id}
                  </p>
                </div>
              </div>

              <div className="flex justify-end space-x-3">
                <Button
                  variant="outline"
                  onClick={() => handleRemoveEntry(selectedEntry.id)}
                  className="text-red-600 hover:text-red-700 border-red-300 hover:border-red-400"
                >
                  <Trash2 size={16} />
                  Delete Entry
                </Button>
                <Button onClick={() => setSelectedEntry(null)}>Close</Button>
              </div>
            </div>
          </Modal>
        )}
      </div>
    </div>
  );
}

export default withAuth(HistoryPage);
