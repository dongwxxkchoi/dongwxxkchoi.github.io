const { Client } = require("@notionhq/client");
const { NotionToMarkdown } = require("notion-to-md");
const moment = require("moment");
const path = require("path");
const fs = require("fs");
const axios = require("axios");
// or
// import {NotionToMarkdown} from "notion-to-md";

const notion = new Client({
  auth: process.env.NOTION_TOKEN,
});

function escapeCodeBlock(body) {
  const regex = /```([\s\S]*?)```/g;
  return body.replace(regex, function (match, htmlBlock) {
    return "\n{% raw %}\n```" + htmlBlock.trim() + "\n```\n{% endraw %}\n";
  });
}

function replaceTitleOutsideRawBlocks(body) {
  const rawBlocks = [];
  const placeholder = "%%RAW_BLOCK%%";
  body = body.replace(/{% raw %}[\s\S]*?{% endraw %}/g, (match) => {
    rawBlocks.push(match);
    return placeholder;
  });

  const regex = /\n#[^\n]+\n/g;
  body = body.replace(regex, function (match) {
    return "\n" + match.replace("\n#", "\n##");
  });

  rawBlocks.forEach(block => {
    body = body.replace(placeholder, block);
  });

  return body;
}

function convertMathBlocks(md) {
  return md.replace(/\\\[(.*?)\\\]/gs, function (match, mathContent) {
    return `$$${mathContent}$$`;
  });
}

// passing notion client to the option
const n2m = new NotionToMarkdown({ notionClient: notion });

(async () => {
  // ensure directory exists
  const root = "_posts";
  fs.mkdirSync(root, { recursive: true });

  const databaseId = process.env.DATABASE_ID;
  let response = await notion.databases.query({
    database_id: databaseId,
    filter: {
      property: "공개",
      checkbox: {
        equals: true,
      },
    },
  });

  const pages = response.results;
  while (response.has_more) {
    const nextCursor = response.next_cursor;
    response = await notion.databases.query({
      database_id: databaseId,
      start_cursor: nextCursor,
      filter: {
        property: "공개",
        checkbox: {
          equals: true,
        },
      },
    });
    pages.push(...response.results);
  }

  const processedFiles = new Set();
  for (const r of pages) {
    const id = r.id;

    const shouldUpdate = r.properties?.["업데이트필요"]?.["checkbox"] === true;
    if (!shouldUpdate) {
      console.log(`Skipping ${id} because "업데이트필요" is not checked.`);
      continue;
    }

    // date
    let date = moment(r.created_time).format("YYYY-MM-DD");
    let pdate = r.properties?.["날짜"]?.["date"]?.["start"];
    if (pdate) {
      date = moment(pdate).format("YYYY-MM-DD");
    }
    // title
    let title = id;
    let ptitle = r.properties?.["게시물"]?.["title"];
    if (ptitle?.length > 0) {
      title = ptitle[0]?.["plain_text"];
    }
    // tags
    let tags = [];
    let ptags = r.properties?.["태그"]?.["multi_select"];
    for (const t of ptags) {
      const n = t?.["name"];
      if (n) {
        tags.push(n);
      }
    }
    // categories
    let cats = [];
    let pcats = r.properties?.["카테고리"]?.["multi_select"];
    for (const t of pcats) {
      const n = t?.["name"];
      if (n) {
        cats.push(n);
      }
    }

    // frontmatter
    let fmtags = "";
    let fmcats = "";
    if (tags.length > 0) {
      fmtags += "\ntags: [";
      for (const t of tags) {
        fmtags += t + ", ";
      }
      fmtags += "]";
    }
    if (cats.length > 0) {
      fmcats += "\ncategories: [";
      for (const t of cats) {
        fmcats += t + ", ";
      }
      fmcats += "]";
    }
    const fm = `---
layout: single
date: ${date}
title: "${title}"
use_math: true
author_profile: false${fmtags}${fmcats}
---
`;
    const mdblocks = await n2m.pageToMarkdown(id);
    let md = n2m.toMarkdownString(mdblocks)["parent"];
    if (md === "") {
      continue;
    }
    
    md = convertMathBlocks(md);
    md = escapeCodeBlock(md);
    md = replaceTitleOutsideRawBlocks(md);

    const ftitle = `${date}-${title.replaceAll(" ", "-")}.md`;
    const filePath = path.join(root, ftitle);

    processedFiles.add(ftitle);

    
    
    let index = 0;
    let edited_md = md.replace(
      /!\[(.*?)\]\((.*?)\)/g,
      function (match, p1, p2, p3) {
        const dirname = path.join("assets/img", ftitle);
        if (!fs.existsSync(dirname)) {
          fs.mkdirSync(dirname, { recursive: true });
        }
        const filename = path.join(dirname, `${index}.png`);
        
        axios({
          method: "get",
          url: p2,
          responseType: "stream",
        })
        .then(function (response) {
          let file = fs.createWriteStream(`${filename}`);
          response.data.pipe(file);
        })
        .catch(function (error) {
          console.log(error);
        });
        
        let res;
        if (p1 === "") res = "";
        else res = `_${p1}_`;
        
        return `![${index++}](/${filename})${res}`;
      }
    );

    let newContent = fm + md;
    if (fs.existsSync(filePath)) {
      const currentContent = fs.readFileSync(filePath, "utf-8");
      if (currentContent === newContent) {
        console.log(`No changes for ${ftitle}. Skipping update.`);
        continue;
      }
    }

    fs.writeFile(path.join(root, ftitle), fm + edited_md, (err) => {
      if (err) {
        console.log(err);
      }
    });
  }

  const existingFiles = fs.readdirSync(root);
  existingFiles.forEach((file) => {
    if (!processedFiles.has(file)) {
      console.log(`Keeping unprocessed file: ${file}`);
    }
  });
})();
