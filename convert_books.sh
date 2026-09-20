cat << 'EOF' > convert_books.sh
#!/bin/bash
source ~/Marker/marker_env/bin/activate
mkdir -p ~/Documents/Tankönyvprojekt/MD/images

for pdf in ~/Documents/Tankönyvprojekt/PDF/*.pdf; do
  [ -e "$pdf" ] || continue
  konyvnev=$(basename "$pdf" .pdf)
  echo "Feldolgozás alatt: $konyvnev"
  marker_single "$pdf" --output_dir ~/Documents/Tankönyvprojekt/MD
  mv ~/Documents/Tankönyvprojekt/MD/"$konyvnev"/"$konyvnev".md ~/Documents/Tankönyvprojekt/MD/
  mkdir -p ~/Documents/Tankönyvprojekt/MD/images/"$konyvnev"
  mv ~/Documents/Tankönyvprojekt/MD/"$konyvnev"/*.png ~/Documents/Tankönyvprojekt/MD/images/"$konyvnev"/ 2>/dev/null
  mv ~/Documents/Tankönyvprojekt/MD/"$konyvnev"/*.jpg ~/Documents/Tankönyvprojekt/MD/images/"$konyvnev"/ 2>/dev/null
  mv ~/Documents/Tankönyvprojekt/MD/"$konyvnev"/*.jpeg ~/Documents/Tankönyvprojekt/MD/images/"$konyvnev"/ 2>/dev/null
  sed -i "s|](_page_|](images/$konyvnev/_page_|g" ~/Documents/Tankönyvprojekt/MD/"$konyvnev".md
  sed -i "s|](image|](images/$konyvnev/image|g" ~/Documents/Tankönyvprojekt/MD/"$konyvnev".md
  rm -rf ~/Documents/Tankönyvprojekt/MD/"$konyvnev"
done
EOF
chmod +x convert_books.sh

