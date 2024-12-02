mkdir 0000.empty_project
pnpm init 
pnpm install --save-dev electron
# add main.js index.html
# 修改 package.json 的 main 字段为 "main": "main.js" start 字段为 "start": "electron ." 
pnpm start

git clone git@github.com:electron/electron-quick-start.git 0000.electron-quick-start