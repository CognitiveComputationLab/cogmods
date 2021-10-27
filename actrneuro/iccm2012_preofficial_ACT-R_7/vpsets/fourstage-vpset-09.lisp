(defparameter *term-presentations* (list
'(("Z" "r" "N") ("N" "l" "K") ("K" "l" "X") ("K" "r" "Z") ("N" "Z" "K" "X") 203 1 "1" 1)
'(("N" "r" "X") ("Z" "r" "X") ("Z" "l" "K") ("N" "l" "X") ("X" "K" "Z" "N") 280 0 "2" 0)
'(("N" "r" "K") ("X" "r" "N") ("X" "l" "Z") ("Z" "r" "K") ("K" "X" "Z" "N") 92 1 "0" 0)
'(("K" "r" "N") ("K" "l" "X") ("X" "l" "Z") ("Z" "l" "K") ("N" "K" "X" "Z") 130 0 "0" 1)
'(("N" "r" "X") ("X" "l" "Z") ("Z" "l" "K") ("N" "r" "Z") ("X" "Z" "N" "K") 251 1 "2" 1)
'(("K" "r" "N") ("X" "r" "N") ("X" "l" "Z") ("K" "l" "X") ("N" "K" "Z" "X") 200 1 "1" 0)
'(("K" "l" "N") ("K" "l" "X") ("X" "l" "Z") ("N" "l" "K") ("K" "N" "Z" "X") 226 0 "1" 0)
'(("X" "l" "Z") ("Z" "l" "K") ("K" "l" "N") ("Z" "l" "N") ("X" "Z" "K" "N") 33 1 "0" 1)
'(("Z" "l" "N") ("Z" "l" "X") ("X" "l" "K") ("K" "l" "N") ("Z" "X" "K" "N") 289 1 "3" 1)
'(("N" "l" "K") ("Z" "r" "N") ("Z" "l" "X") ("N" "r" "X") ("N" "X" "K" "Z") 318 0 "3" 0)
'(("Z" "l" "N") ("X" "r" "Z") ("X" "l" "K") ("Z" "r" "K") ("Z" "N" "X" "K") 221 0 "1" 1)
'(("K" "l" "X") ("Z" "r" "X") ("Z" "l" "N") ("K" "r" "Z") ("K" "Z" "X" "N") 119 0 "0" 0)
'(("Z" "l" "N") ("Z" "l" "K") ("K" "l" "X") ("K" "l" "N") ("Z" "X" "K" "N") 242 1 "2" 0)
'(("N" "l" "Z") ("X" "r" "Z") ("X" "l" "K") ("K" "l" "N") ("N" "X" "K" "Z") 171 0 "0" 0)
'(("Z" "l" "X") ("N" "r" "Z") ("N" "l" "K") ("N" "r" "X") ("Z" "X" "N" "K") 205 1 "1" 1)
'(("Z" "l" "X") ("Z" "l" "N") ("N" "l" "K") ("X" "l" "N") ("Z" "X" "N" "K") 193 1 "1" 1)
'(("Z" "r" "X") ("N" "r" "X") ("N" "l" "K") ("K" "l" "X") ("X" "N" "K" "Z") 311 0 "3" 1)
'(("X" "r" "K") ("X" "l" "Z") ("Z" "l" "N") ("X" "r" "N") ("K" "X" "Z" "N") 146 0 "0" 1)
'(("N" "r" "Z") ("Z" "l" "K") ("K" "l" "X") ("X" "l" "Z") ("Z" "N" "X" "K") 212 0 "1" 0)
'(("K" "r" "Z") ("Z" "l" "X") ("X" "l" "N") ("K" "r" "N") ("Z" "N" "K" "X") 300 1 "3" 0)
'(("X" "r" "N") ("K" "r" "X") ("K" "l" "Z") ("N" "l" "K") ("N" "K" "X" "Z") 8 1 "0" 0)
'(("X" "l" "N") ("N" "l" "K") ("K" "l" "Z") ("Z" "r" "X") ("X" "N" "K" "Z") 81 1 "0" 1)
'(("Z" "r" "X") ("N" "r" "X") ("N" "l" "K") ("X" "r" "Z") ("X" "Z" "K" "N") 240 0 "1" 0)
'(("X" "l" "K") ("N" "r" "X") ("N" "l" "Z") ("X" "r" "K") ("X" "N" "K" "Z") 285 0 "2" 1)
'(("K" "r" "Z") ("N" "r" "Z") ("N" "l" "X") ("K" "l" "Z") ("Z" "K" "X" "N") 232 0 "1" 0)
'(("N" "r" "K") ("Z" "r" "N") ("Z" "l" "X") ("X" "l" "K") ("K" "N" "Z" "X") 164 0 "0" 1)
'(("Z" "l" "X") ("K" "r" "Z") ("K" "l" "N") ("X" "r" "K") ("Z" "K" "X" "N") 253 1 "2" 1)
'(("K" "l" "Z") ("X" "r" "Z") ("X" "l" "N") ("X" "r" "K") ("K" "X" "N" "Z") 27 1 "0" 0)
'(("N" "l" "X") ("Z" "r" "N") ("Z" "l" "K") ("X" "l" "Z") ("N" "X" "Z" "K") 197 1 "1" 1)
'(("X" "r" "Z") ("Z" "l" "K") ("K" "l" "N") ("Z" "r" "N") ("Z" "N" "X" "K") 316 0 "3" 0)
'(("N" "l" "X") ("X" "l" "K") ("K" "l" "Z") ("X" "r" "Z") ("N" "X" "K" "Z") 145 0 "0" 1)
'(("N" "r" "K") ("K" "l" "Z") ("Z" "l" "X") ("K" "r" "X") ("K" "N" "X" "Z") 220 0 "1" 0)
'(("Z" "r" "K") ("X" "r" "K") ("X" "l" "N") ("N" "l" "Z") ("K" "X" "N" "Z") 295 1 "3" 1)
'(("K" "l" "X") ("K" "l" "N") ("N" "l" "Z") ("X" "l" "K") ("K" "Z" "N" "X") 274 0 "2" 0)
'(("K" "r" "N") ("K" "l" "Z") ("Z" "l" "X") ("K" "l" "X") ("N" "Z" "K" "X") 38 1 "0" 0)
'(("N" "l" "Z") ("N" "l" "K") ("K" "l" "X") ("K" "r" "Z") ("N" "Z" "K" "X") 201 1 "1" 1)
'(("K" "r" "Z") ("K" "l" "X") ("X" "l" "N") ("N" "r" "Z") ("Z" "X" "K" "N") 86 1 "0" 0)
'(("N" "l" "X") ("N" "l" "Z") ("Z" "l" "K") ("X" "r" "Z") ("N" "Z" "X" "K") 249 1 "2" 1)
'(("Z" "r" "X") ("K" "r" "X") ("K" "l" "N") ("K" "r" "Z") ("X" "Z" "N" "K") 208 1 "1" 0)
'(("K" "r" "X") ("X" "l" "Z") ("Z" "l" "N") ("N" "l" "K") ("X" "Z" "N" "K") 291 1 "3" 1)
'(("Z" "l" "K") ("N" "r" "K") ("N" "l" "X") ("Z" "l" "N") ("Z" "K" "N" "X") 3 1 "0" 1)
'(("X" "l" "K") ("Z" "r" "X") ("Z" "l" "N") ("N" "l" "X") ("X" "K" "Z" "N") 213 0 "1" 1)
'(("Z" "r" "N") ("X" "r" "N") ("X" "l" "K") ("N" "r" "K") ("N" "K" "Z" "X") 320 0 "3" 0)
'(("X" "l" "N") ("X" "l" "K") ("K" "l" "Z") ("X" "r" "N") ("X" "N" "K" "Z") 233 0 "1" 1)
'(("Z" "r" "K") ("N" "r" "Z") ("N" "l" "X") ("K" "r" "N") ("K" "Z" "N" "X") 116 0 "0" 1)
'(("K" "r" "X") ("X" "l" "N") ("N" "l" "Z") ("K" "l" "N") ("X" "K" "Z" "N") 196 1 "1" 0)
'(("K" "l" "Z") ("Z" "l" "N") ("N" "l" "X") ("X" "l" "K") ("K" "N" "X" "Z") 169 0 "0" 0)
'(("K" "l" "X") ("N" "r" "K") ("N" "l" "Z") ("X" "l" "K") ("K" "Z" "N" "X") 278 0 "2" 0)
))