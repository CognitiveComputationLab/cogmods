(defparameter *term-presentations* (list
'(("N" "l" "X") ("K" "r" "N") ("K" "l" "Z") ("K" "l" "X") ("N" "Z" "K" "X") 246 1 "2" 0)
'(("Z" "l" "X") ("X" "l" "K") ("K" "l" "N") ("N" "r" "X") ("Z" "K" "X" "N") 53 1 "0" 0)
'(("N" "r" "X") ("X" "l" "K") ("K" "l" "Z") ("N" "l" "K") ("X" "N" "K" "Z") 195 1 "1" 1)
'(("N" "r" "Z") ("X" "r" "Z") ("X" "l" "K") ("N" "r" "K") ("Z" "K" "N" "X") 304 1 "3" 0)
'(("K" "r" "Z") ("X" "r" "Z") ("X" "l" "N") ("N" "l" "Z") ("Z" "K" "X" "N") 215 0 "1" 1)
'(("Z" "r" "X") ("N" "r" "Z") ("N" "l" "K") ("K" "l" "X") ("X" "Z" "N" "K") 164 0 "0" 1)
'(("X" "r" "Z") ("X" "l" "N") ("N" "l" "K") ("X" "r" "K") ("Z" "X" "N" "K") 146 0 "0" 1)
'(("Z" "l" "K") ("N" "r" "Z") ("N" "l" "X") ("Z" "r" "K") ("Z" "K" "X" "N") 238 0 "1" 0)
'(("N" "r" "K") ("K" "l" "Z") ("Z" "l" "X") ("N" "l" "K") ("K" "Z" "X" "N") 323 0 "3" 1)
'(("K" "l" "X") ("K" "l" "Z") ("Z" "l" "N") ("Z" "r" "X") ("K" "X" "N" "Z") 202 1 "1" 0)
'(("N" "l" "K") ("X" "r" "K") ("X" "l" "Z") ("N" "l" "Z") ("N" "X" "Z" "K") 75 1 "0" 0)
'(("K" "l" "Z") ("K" "l" "N") ("N" "l" "X") ("K" "r" "X") ("K" "N" "Z" "X") 265 0 "2" 1)
'(("Z" "r" "N") ("X" "r" "Z") ("X" "l" "K") ("N" "l" "X") ("N" "Z" "X" "K") 4 1 "0" 1)
'(("Z" "r" "N") ("K" "r" "N") ("K" "l" "X") ("Z" "l" "K") ("N" "Z" "K" "X") 199 1 "1" 1)
'(("N" "l" "X") ("Z" "r" "N") ("Z" "l" "K") ("K" "l" "N") ("N" "K" "Z" "X") 262 0 "2" 0)
'(("N" "r" "X") ("X" "l" "K") ("K" "l" "Z") ("X" "r" "Z") ("X" "N" "K" "Z") 219 0 "1" 1)
'(("K" "r" "X") ("K" "l" "N") ("N" "l" "Z") ("Z" "r" "K") ("X" "N" "K" "Z") 54 1 "0" 0)
'(("N" "r" "Z") ("K" "r" "Z") ("K" "l" "X") ("N" "r" "K") ("Z" "K" "N" "X") 255 1 "2" 1)
'(("Z" "l" "X") ("N" "r" "Z") ("N" "l" "K") ("N" "r" "X") ("Z" "X" "K" "N") 206 1 "1" 0)
'(("Z" "l" "K") ("X" "r" "K") ("X" "l" "N") ("X" "l" "Z") ("Z" "K" "X" "N") 99 0 "0" 1)
'(("X" "l" "Z") ("X" "l" "K") ("K" "l" "N") ("X" "r" "Z") ("X" "K" "N" "Z") 329 0 "3" 1)
'(("Z" "r" "X") ("X" "l" "N") ("N" "l" "K") ("K" "l" "Z") ("X" "K" "Z" "N") 292 1 "3" 0)
'(("K" "l" "Z") ("Z" "l" "X") ("X" "l" "N") ("K" "r" "N") ("K" "X" "N" "Z") 185 0 "0" 0)
'(("X" "l" "N") ("X" "l" "Z") ("Z" "l" "K") ("N" "l" "X") ("X" "N" "K" "Z") 226 0 "1" 0)
'(("Z" "l" "X") ("Z" "l" "K") ("K" "l" "N") ("Z" "r" "X") ("Z" "X" "K" "N") 233 0 "1" 1)
'(("K" "r" "X") ("X" "l" "Z") ("Z" "l" "N") ("K" "r" "Z") ("X" "N" "Z" "K") 252 1 "2" 0)
'(("X" "r" "K") ("X" "l" "N") ("N" "l" "Z") ("K" "l" "N") ("K" "X" "N" "Z") 2 1 "0" 1)
'(("N" "r" "K") ("Z" "r" "K") ("Z" "l" "X") ("X" "l" "N") ("K" "Z" "X" "N") 295 1 "3" 1)
'(("N" "l" "K") ("K" "l" "X") ("X" "l" "Z") ("Z" "l" "K") ("N" "K" "X" "Z") 129 0 "0" 1)
'(("N" "l" "Z") ("X" "r" "N") ("X" "l" "K") ("Z" "l" "X") ("N" "Z" "K" "X") 198 1 "1" 0)
'(("K" "l" "X") ("Z" "r" "X") ("Z" "l" "N") ("N" "r" "X") ("K" "Z" "X" "N") 55 1 "0" 0)
'(("K" "l" "N") ("K" "l" "X") ("X" "l" "Z") ("Z" "l" "K") ("K" "X" "N" "Z") 257 0 "2" 1)
'(("N" "r" "X") ("X" "l" "Z") ("Z" "l" "K") ("K" "l" "X") ("X" "N" "K" "Z") 212 0 "1" 0)
'(("Z" "r" "X") ("K" "r" "Z") ("K" "l" "N") ("X" "r" "N") ("X" "K" "N" "Z") 188 0 "0" 0)
'(("K" "r" "X") ("N" "r" "X") ("N" "l" "Z") ("N" "r" "K") ("X" "K" "N" "Z") 207 1 "1" 1)
'(("N" "l" "X") ("Z" "r" "N") ("Z" "l" "K") ("N" "r" "X") ("N" "K" "X" "Z") 334 0 "3" 0)
'(("K" "l" "Z") ("X" "r" "K") ("X" "l" "N") ("K" "r" "N") ("K" "X" "Z" "N") 269 0 "2" 1)
'(("K" "r" "Z") ("N" "r" "Z") ("N" "l" "X") ("Z" "r" "X") ("Z" "K" "X" "N") 224 0 "1" 0)
'(("N" "r" "Z") ("X" "r" "N") ("X" "l" "K") ("K" "r" "Z") ("Z" "N" "X" "K") 84 1 "0" 1)
'(("K" "l" "X") ("K" "l" "N") ("N" "l" "Z") ("N" "l" "X") ("K" "Z" "N" "X") 242 1 "2" 0)
'(("N" "l" "K") ("Z" "r" "N") ("Z" "l" "X") ("K" "l" "N") ("N" "K" "Z" "X") 229 0 "1" 1)
'(("Z" "l" "N") ("K" "r" "N") ("K" "l" "X") ("N" "r" "X") ("Z" "N" "K" "X") 147 0 "0" 1)
'(("Z" "r" "N") ("N" "l" "K") ("K" "l" "X") ("Z" "r" "X") ("N" "K" "X" "Z") 299 1 "3" 1)
'(("X" "l" "K") ("K" "l" "Z") ("Z" "l" "N") ("X" "l" "Z") ("X" "Z" "N" "K") 9 1 "0" 0)
'(("K" "l" "Z") ("K" "l" "N") ("N" "l" "X") ("Z" "l" "N") ("K" "Z" "X" "N") 194 1 "1" 0)
'(("N" "r" "K") ("X" "r" "K") ("X" "l" "Z") ("N" "l" "K") ("K" "Z" "N" "X") 328 0 "3" 0)
'(("K" "r" "N") ("K" "l" "Z") ("Z" "l" "X") ("X" "l" "N") ("N" "Z" "K" "X") 166 0 "0" 0)
'(("X" "r" "N") ("N" "l" "K") ("K" "l" "Z") ("K" "r" "X") ("N" "X" "K" "Z") 203 1 "1" 1)
))