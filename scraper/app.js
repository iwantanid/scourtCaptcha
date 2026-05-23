const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const SAVE_DIR = path.join(__dirname, "downloaded");

if (!fs.existsSync(SAVE_DIR)) {
    fs.mkdirSync(SAVE_DIR);
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

// 사용법:
// node app.js [시작번호] [반복횟수]
//
// 예:
// node app.js 1000 500

const startIndex = parseInt(process.argv[2] || "", 10);
const repeatCount = parseInt(process.argv[3] || "", 10);

if (
    isNaN(startIndex) ||
    isNaN(repeatCount) ||
    startIndex < 0 ||
    repeatCount <= 0
) {
    console.log("usage: node app.js [start_number] [repeat_count]");
    process.exit(1);
}

(async () => {

    const browser = await chromium.launch({
        headless: false
    });

    const page = await browser.newPage();

    for (let n = 0; n < repeatCount; n++) {

        const currentIndex = startIndex + n;

        console.log(`[${currentIndex}] loading...`);

        await page.goto(
            "https://www.scourt.go.kr/portal/information/events/search/search.jsp",
            {
                waitUntil: "networkidle"
            }
        );

        // iframe 접근
        const frame = page.frames()[1];

        if (!frame) {

            console.log("frame not found");

            continue;
        }

        // captcha 이미지
        const captcha = frame.locator(
            "#mf_ssgoTopMainTab_contents_content1_body_img_captcha"
        );

        // captcha 로딩 대기
        await captcha.waitFor({
            timeout: 10000
        });

        // 파일명
        const filename =
            `${String(currentIndex).padStart(4, "0")}.png`;

        const filepath =
            path.join(SAVE_DIR, filename);

        // 저장
        await captcha.screenshot({
            path: filepath
        });

        console.log(`saved: ${filepath}`);

        // 1~2초 랜덤 대기
        await sleep(
            1000 + Math.random() * 1000
        );
    }

    await browser.close();

})();