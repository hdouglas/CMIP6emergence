library(cartogram)
library(dplyr)
library(tmap)
library(tmaptools)
library(readr)
library(rgdal)
library(ncdf4)
library(ncdf4.helpers)
library(ggplot2)

##------------- Prepare the data

# Read shapefile. 
ne50m <- readOGR(dsn="/Users/hunterdouglas/Documents/VUW/Exercises/Geographies/ne_50m_admin_0_countries",
                 layer="ne_50m_admin_0_countries")

# Population data
pop119 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countryPop_ssp119_2040-2060.csv",",")
pop126 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countryPop_ssp126_2040-2060.csv",",")
pop245 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countryPop_ssp245_2040-2060.csv",",")
pop370 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countryPop_ssp370_2040-2060.csv",",")
pop585 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countryPop_ssp585_2040-2060.csv",",")

# S/N data
SN_16_ssp119 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_16_ssp119_2040-2060.csv",",")
SN_16_ssp126 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_16_ssp126_2040-2060.csv",",")
SN_16_ssp245 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_16_ssp245_2040-2060.csv",",")
SN_16_ssp370 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_16_ssp370_2040-2060.csv",",")
SN_16_ssp585 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_16_ssp585_2040-2060.csv",",")
SN_50_ssp119 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_50_ssp119_2040-2060.csv",",")
SN_50_ssp126 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_50_ssp126_2040-2060.csv",",")
SN_50_ssp245 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_50_ssp245_2040-2060.csv",",")
SN_50_ssp370 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_50_ssp370_2040-2060.csv",",")
SN_50_ssp585 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_50_ssp585_2040-2060.csv",",")
SN_84_ssp119 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_84_ssp119_2040-2060.csv",",")
SN_84_ssp126 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_84_ssp126_2040-2060.csv",",")
SN_84_ssp245 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_84_ssp245_2040-2060.csv",",")
SN_84_ssp370 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_84_ssp370_2040-2060.csv",",")
SN_84_ssp585 = read_delim("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/CSVs/countrySN_84_ssp585_2040-2060.csv",",")

# Do a tabular join using dplyr on country codes
ne50m@data <- left_join(ne50m@data,pop119,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,pop126,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,pop245,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,pop370,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,pop585,by=c("ADM0_A3" = "Country"))

ne50m@data <- left_join(ne50m@data,SN_16_ssp119,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,SN_16_ssp126,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,SN_16_ssp245,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,SN_16_ssp370,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,SN_16_ssp585,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,SN_50_ssp119,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,SN_50_ssp126,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,SN_50_ssp245,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,SN_50_ssp370,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,SN_50_ssp585,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,SN_84_ssp119,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,SN_84_ssp126,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,SN_84_ssp245,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,SN_84_ssp370,by=c("ADM0_A3" = "Country"))
ne50m@data <- left_join(ne50m@data,SN_84_ssp585,by=c("ADM0_A3" = "Country"))

# assign a projection (
population_map <- spTransform(ne50m, CRS("+init=epsg:3857")) #mercator
population_map <- spTransform(ne50m, CRS("+init=esri:54030")) #robinson

# # create a nice bounding box for the EU countries
# bb_cont_eu <- bb(population_map, 
#                  xlim = c(0.58, 0.96), 
#                  ylim = c(0.09, 0.82), 
#                  relative=TRUE)

# cartogram doesn"t like NA values so fill them with zero
population_map@data["Population (millions)_ssp119"] [is.na(population_map@data["Population (millions)_ssp119"] )] <- 0
population_map@data["Population (millions)_ssp126"] [is.na(population_map@data["Population (millions)_ssp126"] )] <- 0
population_map@data["Population (millions)_ssp245"] [is.na(population_map@data["Population (millions)_ssp245"] )] <- 0
population_map@data["Population (millions)_ssp370"] [is.na(population_map@data["Population (millions)_ssp370"] )] <- 0
population_map@data["Population (millions)_ssp585"] [is.na(population_map@data["Population (millions)_ssp585"] )] <- 0

##------------- CONTIGUOUS AREA CARTOGRAM

# Generate the contiguous area cartogram, this might take a moment to run through all the 12 iterations
# You may notice with different maps and datasets that you will need to play around with the number of iterations and the threshold to succesfully plot a cartogram
im = 12
th = 0.315
population_map_119 <- cartogram_cont(population_map, "Population (millions)_ssp119", itermax = im, prepare = "adjust", threshold = th)
population_map_126 <- cartogram_cont(population_map, "Population (millions)_ssp126", itermax = im, prepare = "adjust", threshold = th)
population_map_245 <- cartogram_cont(population_map, "Population (millions)_ssp245", itermax = im, prepare = "adjust", threshold = th)
population_map_370 <- cartogram_cont(population_map, "Population (millions)_ssp370", itermax = im, prepare = "adjust", threshold = th)
population_map_585 <- cartogram_cont(population_map, "Population (millions)_ssp585", itermax = im, prepare = "adjust", threshold = th)

#Plotting the continuous cartogram
cbreaks = c(0,1.6,2.2,2.8,3.6,4.4,5.4,6.7,9.0,30.0)
cbreaks = c(0.0,1.0,2.0,3.0,5.0,15.3)

myPalette = c("#FFFFFF","#FFFEC7","#FCD287","#E67474","#B13E3E")

fn119 = paste("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/cartogram_ssp119_",im,"-",th,"_2040-2060.svg",sep="")
#jpeg(file=fn119, width=1000, height=600)
svg(file=fn119)
tm_shape(population_map_119) + 
  tm_polygons("S.N_50_ssp119", palette = palette(myPalette), style = "fixed", breaks=cbreaks, legend.show = FALSE, border.col = "black", lwd = 0.2) +
  tm_layout(frame = FALSE,
            inner.margins=c(.02,.02,.02,.02))
dev.off()

fn126 = paste("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/cartogram_ssp126_",im,"-",th,"_2040-2060.svg",sep="")
#jpeg(file=fn126, width=1000, height=600)
svg(file=fn126)
tm_shape(population_map_126) + 
  tm_polygons("S.N_50_ssp126", palette = palette(myPalette), style = "fixed", breaks=cbreaks, legend.show = FALSE, border.col = "black", lwd = 0.2) +
  tm_layout(frame = FALSE,
            inner.margins=c(.02,.02,.02,.02))
dev.off()

fn245 = paste("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/cartogram_ssp245_",im,"-",th,"_2040-2060.svg",sep="")
#jpeg(file=fn245, width=1000, height=600)
svg(file=fn245)
tm_shape(population_map_245) + 
  tm_polygons("S.N_50_ssp245", palette = palette(myPalette), style = "fixed", breaks=cbreaks, legend.show = FALSE, border.col = "black", lwd = 0.2) +
  tm_layout(frame = FALSE,
            inner.margins=c(.02,.02,.02,.02))
dev.off()

fn370 = paste("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/cartogram_ssp370_",im,"-",th,"_2040-2060.svg",sep="")
#jpeg(file=fn370, width=1000, height=600)
svg(file=fn370)
tm_shape(population_map_370) + 
  tm_polygons("S.N_50_ssp370", palette = palette(myPalette), style = "fixed", breaks=cbreaks, legend.show = FALSE, border.col = "black", lwd = 0.2) +
  tm_layout(frame = FALSE,
            inner.margins=c(.02,.02,.02,.02))
dev.off()

fn585 = paste("/Users/hunterdouglas/Documents/VUW/Exercises/caRtogram/cartogram_ssp585_",im,"-",th,"_2040-2060.svg",sep="")
#jpeg(file=fn585, width=1000, height=600)
svg(file=fn585)
tm_shape(population_map_585) + 
  tm_polygons("S.N_50_ssp585", palette = palette(myPalette), style = "fixed", breaks=cbreaks, legend.show = FALSE, border.col = "black", lwd = 0.2) +
  tm_layout(frame = FALSE,
            inner.margins=c(.02,.02,.02,.02))
dev.off()


