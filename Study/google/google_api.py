api_key = 'AIzaSyAN-QsR4SVnpX-hQzMCEZsyB8RlFLraSMs'

########################################################################################################################
# 구글 지오코드 API
# https://maps.googleapis.com/maps/api/geocode/json?address=서울 강남구 테헤란로 123&key=AIzaSyAN-QsR4SVnpX-hQzMCEZsyB8RlFLraSMs
# lat = 37.499516, lng = 127.0313815
########################################################################################################################
# {
#    "results" : [
#       {
#          "address_components" : [
#             {
#                "long_name" : "123",
#                "short_name" : "123",
#                "types" : [ "political", "premise" ]
#             },
#             {
#                "long_name" : "테헤란로",
#                "short_name" : "테헤란로",
#                "types" : [ "political", "sublocality", "sublocality_level_4" ]
#             },
#             {
#                "long_name" : "강남구",
#                "short_name" : "강남구",
#                "types" : [ "political", "sublocality", "sublocality_level_1" ]
#             },
#             {
#                "long_name" : "서울특별시",
#                "short_name" : "서울특별시",
#                "types" : [ "locality", "political" ]
#             },
#             {
#                "long_name" : "대한민국",
#                "short_name" : "KR",
#                "types" : [ "country", "political" ]
#             },
#             {
#                "long_name" : "135-081",
#                "short_name" : "135-081",
#                "types" : [ "postal_code" ]
#             }
#          ],
#          "formatted_address" : "대한민국 서울특별시 강남구 테헤란로 123",
#          "geometry" : {
#             "location" : {
#                "lat" : 37.499516,
#                "lng" : 127.0313815
#             },
#             "location_type" : "ROOFTOP",
#             "viewport" : {
#                "northeast" : {
#                   "lat" : 37.5008649802915,
#                   "lng" : 127.0327304802915
#                },
#                "southwest" : {
#                   "lat" : 37.4981670197085,
#                   "lng" : 127.0300325197085
#                }
#             }
#          },
#          "place_id" : "ChIJR0xyt1ehfDUR61CHldf58nU",
#          "types" : [ "political", "premise" ]
#       }
#    ],
#    "status" : "OK"
# }


########################################################################################################################
# 구글 타임존 API
# https://maps.googleapis.com/maps/api/timezone/json?location=37.499516,127.0313815&timestamp=1490093834&key=AIzaSyAN-QsR4SVnpX-hQzMCEZsyB8RlFLraSMs
########################################################################################################################
# {
#    "dstOffset" : 0,
#    "rawOffset" : 32400,
#    "status" : "OK",
#    "timeZoneId" : "Asia/Seoul",
#    "timeZoneName" : "Korean Standard Time"
# }


########################################################################################################################
# 구글 고도 API
# https://maps.googleapis.com/maps/api/elevation/json?locations=37.499516,127.0313815&key=AIzaSyAN-QsR4SVnpX-hQzMCEZsyB8RlFLraSMs
########################################################################################################################
# {
#    "results" : [
#       {
#          "elevation" : 47.80671691894531,
#          "location" : {
#             "lat" : 37.499516,
#             "lng" : 127.0313815
#          },
#          "resolution" : 152.7032318115234
#       }
#    ],
#    "status" : "OK"
# }
'''
import json
from urllib.request import urlopen

def getContry(ipAddress):
    response = urlopen('http://freegeoip.net/json/'+ipAddress).read().decode('utf-8')
    responseJson = json.loads(response)
    return responseJson.get('country_code')

print(getContry('50.78.253.58'))
'''

import json

jsonString = '''{"arrayOfNums":[{"number":0}, {"number":1}, {"number":2}],
                 "arrayOfFruits":[{"fruit":"apple"}, {"fruit":"banana"},{"fruit":"pear"}]}'''
jsonObj = json.loads(jsonString)

print(jsonObj.get('arrayOfNums'))
print(jsonObj.get('arrayOfNums')[1])
print(jsonObj.get('arrayOfNums')[1].get('number')+jsonObj.get('arrayOfNums')[2].get('number'))
print(jsonObj.get('arrayOfFruits')[2].get('fruit'))
