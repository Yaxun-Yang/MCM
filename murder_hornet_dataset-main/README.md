# Murder Hornet Dataset

## Goal of the project

Annotated dataset for Asian Giant Hornets.

* Labeled with [labelImg](https://github.com/tzutalin/labelImg) as Pascal VOC XML format
* Added YOLO format labels from XML files with [XmlToTxt](https://github.com/Isabek/XmlToTxt)
* Images crawled from Naver using [AutoCrawler](https://github.com/YoongiKim/AutoCrawler)
* Removed duplicate umages using [imagededup](https://idealo.github.io/imagededup/user_guide/finding_duplicates/)
* Image and labeled resized with [resize_dataset_pascalvoc](https://github.com/italojs/resize_dataset_pascalvoc)



## Asian Giant Hornets?

* In 2020, Asian Giant Hornet queens are discovered in U.S 
* 30 Asian Giant Hornets can wipe out more than 30000 Asian honeybees. Asian Giant Hornets scavenges honeycomb, leaves no survivors.
* It is expected that U.S. honeybees will be more vulnerable for Asian Giant Hornets. Asian honeybees have defense mechanism of forming up "bee ball" to commence suicidal attacks, whereas U.S Honeybees don't.
* To protect honeybees from Asian Giant Hornets, beekeepers should be alerted as qucikly as possible. One killer hornet calls for other killer hornets.

### To-do list

- [x] Collect and annotate more than 250+ imgs
- [x] Remove close-range pictures: head only or body only pictures
- [x] Remove duplicate pictures

