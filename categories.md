---
layout: page
permalink: /categories/
title: Categories
---


<div>
{% for category in site.categories %}
  <div>
    {% capture category_name %}{{ category | first }}{% endcapture %}
    <div id="#{{ category_name | slugize }}"></div>
    <p></p>
    
    <h3>{{ category_name | capitalize }}</h3>
    <a name="{{ category_name | slugize }}"></a>
    {% for post in site.categories[category_name] %}
    <article>
        <li><font size="3"><a href="{{ site.baseurl }}{{ post.url }}">{{post.title}}</a></font></li>
    </article>
    {% endfor %}
  </div>
{% endfor %}
</div>