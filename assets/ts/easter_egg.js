export const fileNames = [
	"boo.txt",
	"shape.txt",
	"wrap.txt",
];

const baseDir = "/assets/social/ascii/";

const colour = "#000000";

export async function loadFileConcurrently(fileName) {
	try {
		const content = await fetch(baseDir + fileName).then(
			(response) => {
				if (!response.ok) {
					throw new Error(
						`Failed to fetch ${fileName}: ${response.statusText}`,
					);
				}
				return response.text();
			},
		);
		return content.split("';'").filter((item) => item);
	} catch (error) {
		console.error("Error loading files:", error);
		return undefined;
	}
}

export async function play(n, outputfield) {
	const frames = n.length - 1;
	while (true) {
		for (let i = 0; i < frames; i++) {
			document.getElementById(outputfield).innerHTML =
				'<font color="' + colour + '"><pre>' + n[i] +
				"</font></pre>";
			await new Promise((r) => setTimeout(r, 100));
		}
	}
}
