rm -rf ../deploy && mkdir ../deploy
cp -r * ../deploy

cd ../deploy

git init
git add .
git commit -m "Publish"
git branch -M main
git remote add origin git@github.com:hslu-aai/xai

git push -f origin main
