# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 21:49:31 2016

@author: Kimberly
"""

import media

toy_story = media.Movie(
    "Toy Story",
    "A story of a boy and his toys that come to life",
    "http://upload.wikimedia.org/wikipedia/en/1/13/Toy_Story.jpg",
    "https://www.youtube.com/watch?v=vwyZH85NQC4"
    )

avatar = media.Movie(
    "Avatar",
    "A marine on an alien planet",
    "http://upload.wikimedia.org/wikipedia/id/b/b0/Avatar-Teaser-Poster.jpg",
    "https://www.youtube.com/watch?v=uZNHIU3uHT4"
    )


print toy_story.storyline
print avatar.title
avatar.show_trailer()
#toy_story.show_trailer()