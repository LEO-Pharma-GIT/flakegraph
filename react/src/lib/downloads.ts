/** Hand the browser a file to save; the one place a download is made. */
export function downloadFile(name: string, contents: string, type: string): void {
  const blob = new Blob([contents], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

export function downloadJson(name: string, value: unknown): void {
  downloadFile(name, JSON.stringify(value, null, 2), "application/json");
}
