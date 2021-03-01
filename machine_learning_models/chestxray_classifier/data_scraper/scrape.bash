scrape_all() {
	for((i=1;i<92;i++))
	do
		python scrape_js.py https://www.shutterstock.com/search/x+ray+body+parts?page=${i}
	done
}

scrape_all
